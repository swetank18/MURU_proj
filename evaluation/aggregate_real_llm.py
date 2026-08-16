#!/usr/bin/env python3
"""
aggregate_real_llm.py — Aggregate real-LLM evaluation results into LaTeX tables.

Reads JSON result files from evaluation/baselines/ that match a pattern,
recomputes metrics via the canonical metrics module (so ECE matches the
methodology in the paper), runs the same paired-bootstrap procedure as
bootstrap_analysis.py, and emits LaTeX-ready strings.

Outputs:
  - evaluation/baselines/real_llm_summary.json (machine-readable)
  - stdout: LaTeX snippets ready to paste into the paper

Usage:
  python evaluation/aggregate_real_llm.py
"""

import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import MURUMetrics, Prediction
from evaluation.bootstrap_analysis import (
    ece_from_paired,
    mcnemar_test,
    per_problem_outcomes,
)
from evaluation import calibration_metrics as cm
from evaluation import overconfidence as oc
from evaluation import unit_accounting as ua
from evaluation.parse_status import (
    MODEL_FAILURE_STATUSES,
    classify_entry,
    status_report,
)


N_BOOTSTRAP = 10_000

# Confidence assigned to a response the parser could not read. The prompt asks
# for an explicit CONFIDENCE field; where the model emitted one before the
# response became unreadable we keep it, and otherwise fall back to the same
# 0.5 default the parser uses. Only relevant to the strict accounting.
UNPARSED_CONFIDENCE_DEFAULT = 0.5

# A strict-vs-lenient gap above this is reported as a headline caveat: past it,
# the leaderboard is partly measuring format compliance rather than reasoning.
ACCOUNTING_GAP_THRESHOLD = 0.03

# Map result filename pattern (model slug) to display name.
DISPLAY_NAMES = {
    "llama-3_3-70b": "Llama-3.3-70B",
    "llama-3_1-8b": "Llama-3.1-8B",
    "llama-3_1-70b": "Llama-3.1-70B (now Llama-3.3)",
    "gpt-oss-120b": "GPT-OSS-120B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "qwen3-32b": "Qwen3-32B",
    "llama-4-scout": "Llama-4-Scout-17B",
    "mixtral-8x7b": "Mixtral-8x7B",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "claude-3-5-sonnet": "Claude 3.5 Sonnet",
    "claude-3-5-haiku": "Claude 3.5 Haiku",
    "gemini-1_5-pro": "Gemini 1.5 Pro",
    "gemini-1_5-flash": "Gemini 1.5 Flash",
}


def load_problems(subset_dir: Path):
    problems = []
    for filepath in sorted(subset_dir.rglob("MURU-*.json")):
        with open(filepath) as f:
            problems.append(json.load(f))
    return problems


def parse_predictions_from_raw(raw_results, problems_by_id):
    """Build Prediction objects from a result file's raw_results array."""
    predictions = []
    for r in raw_results:
        if not r.get("success"):
            continue
        pid = r["problem_id"]
        if pid not in problems_by_id:
            continue
        parsed = r.get("parsed", {})
        if parsed.get("point_estimate") is None:
            continue
        predictions.append(Prediction(
            problem_id=pid,
            predicted_answer=parsed["point_estimate"],
            predicted_confidence=parsed.get("confidence", 0.5),
            predicted_interval=tuple(parsed["confidence_interval"]) if parsed.get("confidence_interval") else None,
            predicted_framework=parsed.get("framework"),
            raw_response=r.get("response", ""),
        ))
    return predictions


def unit_normalized(preds, problems_by_id):
    """Predictions restated in the ground truth's units where corroborated.

    Applying the correction here, once, means every downstream statistic --
    accuracy, ECE, per-category breakdowns, interval widths, the interval
    score -- sees a prediction that has been read in the unit the model
    actually used. Predictions that need no correction pass through
    unchanged, so this is the identity on any run made under a prompt that
    states the unit convention (``PROMPT_VERSION`` >= 2).
    """
    out, factors = [], {}
    for p in preds:
        problem = problems_by_id[p.problem_id]
        k = ua.credited_factor(p.predicted_answer, p.predicted_interval, problem)
        factors[p.problem_id] = k
        if k == 1.0:
            out.append(p)
            continue
        interval = p.predicted_interval
        out.append(Prediction(
            problem_id=p.problem_id,
            predicted_answer=p.predicted_answer / k,
            predicted_confidence=p.predicted_confidence,
            predicted_interval=(interval[0] / k, interval[1] / k) if interval else None,
            predicted_framework=p.predicted_framework,
            raw_response=p.raw_response,
        ))
    return out, factors


def unparsed_model_failures(raw_results, problems_by_id):
    """Records the model answered but the harness could not read.

    Returns ``(problem_id, stated_confidence)`` pairs for the statuses that are
    the model's fault. Provider-side failures (endpoint 404s, rate limits,
    timeouts) are deliberately not returned: they are missing data and say
    nothing about the model, so they stay out of both accountings.
    """
    out = []
    for r in raw_results:
        pid = r.get("problem_id")
        if pid not in problems_by_id:
            continue
        if classify_entry(r) not in MODEL_FAILURE_STATUSES:
            continue
        conf = (r.get("parsed") or {}).get("confidence")
        out.append((pid, conf if conf is not None else UNPARSED_CONFIDENCE_DEFAULT))
    return out


def sharpness_records(preds, problems_by_id):
    """Per-problem inputs for the interval-score / width statistics."""
    records = []
    for p in preds:
        gt = problems_by_id[p.problem_id]["ground_truth"]
        lo_hi = p.predicted_interval
        ci = gt["confidence_interval"]
        records.append({
            "lower": lo_hi[0] if lo_hi else None,
            "upper": lo_hi[1] if lo_hi else None,
            "gt_point": gt["point_estimate"],
            "gt_width": ci[1] - ci[0],
            "alpha": 1.0 - gt.get("ci_level", 0.9),
        })
    return records


def find_latest_result(model_slug: str):
    """Find the latest JSON file for a given model slug in evaluation/baselines/."""
    baselines = PROJECT_ROOT / "evaluation" / "baselines"
    matches = sorted(baselines.glob(f"{model_slug}_*.json"))
    if not matches:
        return None
    return matches[-1]


def bootstrap_metrics(correct, fwmatch, overconf, conf, n_boot=N_BOOTSTRAP, seed=0xC0FFEE):
    """Percentile bootstrap CIs on accuracy, ece, overconf, framework_match."""
    rng = random.Random(seed)
    n = len(correct)

    acc_b, fw_b, oc_b, ece_b = [], [], [], []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        c = [correct[i] for i in idxs]
        f = [fwmatch[i] for i in idxs]
        o = [overconf[i] for i in idxs]
        co = [conf[i] for i in idxs]
        acc_b.append(sum(c) / n)
        fw_b.append(sum(f) / n)
        oc_b.append(sum(o) / n)
        ece_b.append(ece_from_paired(c, co))

    def ci(arr):
        s = sorted(arr)
        return s[int(0.025 * n_boot)], s[int(0.975 * n_boot)]

    return {
        "accuracy": {"point": sum(correct) / n, "ci95": ci(acc_b)},
        "ece": {"point": ece_from_paired(correct, conf), "ci95": ci(ece_b)},
        "overconfidence": {"point": sum(overconf) / n, "ci95": ci(oc_b)},
        "framework_match": {"point": sum(fwmatch) / n, "ci95": ci(fw_b)},
    }


def fmt_pct(x, ci):
    return f"{100*x:.1f}\\%~[{100*ci[0]:.1f}, {100*ci[1]:.1f}]"


def fmt_dec(x, ci):
    return f"{x:.3f}~[{ci[0]:.3f}, {ci[1]:.3f}]"


def main():
    test_dir = PROJECT_ROOT / "data" / "test"
    problems = load_problems(test_dir)
    problems_by_id = {p["id"]: p for p in problems}
    print(f"Test set: {len(problems)} problems", file=sys.stderr)

    summary = {}
    for slug, display in DISPLAY_NAMES.items():
        path = find_latest_result(slug)
        if not path:
            continue
        with open(path) as f:
            data = json.load(f)
        raw = data.get("raw_results", [])
        if not raw:
            continue
        preds = parse_predictions_from_raw(raw, problems_by_id)
        if not preds:
            continue

        # Salvaged result files lack predicted_framework, so framework-match
        # rate is structurally zero and should be displayed as "n/a" rather
        # than "0%". Detect by file flag or filename suffix.
        salvaged = bool(data.get("salvaged")) or "salvaged" in path.name

        # Restrict problem set to those that this model actually answered
        # (so the per-problem outcome lists align).
        answered_ids = {p.problem_id for p in preds}
        sub_problems = [p for p in problems if p["id"] in answered_ids]

        c, f, o, conf = per_problem_outcomes(sub_problems, preds)
        boot = bootstrap_metrics(c, f, o, conf)

        # Framework-match: compute only over predictions that actually carry a
        # framework. Entries recovered from progress logs (salvage path) lost
        # the raw response, so their framework is missing data, not a wrong
        # answer — scoring them as 0 would understate a model whose archive
        # mixes fresh and salvaged rows (e.g. an accumulated union). Models
        # with full framework coverage are unchanged (subset == full set).
        fw_hits, fw_seen = [], 0
        for p in preds:
            if p.predicted_framework:
                fw_seen += 1
                fw_hits.append(int(p.predicted_framework == problems_by_id[p.problem_id]["required_framework"]))
        if fw_seen:
            rng = random.Random(0xF00D)
            nfw = len(fw_hits)
            boots = [sum(fw_hits[rng.randrange(nfw)] for _ in range(nfw)) / nfw for _ in range(N_BOOTSTRAP)]
            boots.sort()
            boot["framework_match"] = {
                "point": sum(fw_hits) / nfw,
                "ci95": [boots[int(0.025 * N_BOOTSTRAP)], boots[int(0.975 * N_BOOTSTRAP)]],
                "n": nfw,
            }
        else:
            boot["framework_match"] = {"point": None, "ci95": [None, None], "n": 0}

        # Difficulty / category breakdown via canonical metrics
        m = MURUMetrics(sub_problems, preds)
        breakdown = m.compute_all()

        # ── Parse accounting (P0) ────────────────────────────────
        # Two versions of every headline metric: "lenient" drops responses the
        # harness could not read, "strict" scores them incorrect. The gap is
        # the part of the leaderboard that is format compliance rather than
        # reasoning, and it has to be visible rather than absorbed.
        status = status_report(data, n_test=len(problems), test_ids=set(problems_by_id))
        failures = unparsed_model_failures(raw, problems_by_id)
        if failures:
            c_strict = c + [0] * len(failures)
            f_strict = f + [0] * len(failures)
            o_strict = o + [
                1 if cf > oc.CANONICAL_THRESHOLD else 0 for _, cf in failures
            ]
            conf_strict = conf + [cf for _, cf in failures]
            boot_strict = bootstrap_metrics(c_strict, f_strict, o_strict, conf_strict)
        else:
            boot_strict = boot
        acc_gap = boot["accuracy"]["point"] - boot_strict["accuracy"]["point"]
        ece_gap = boot_strict["ece"]["point"] - boot["ece"]["point"]

        # ── Unit accounting ──────────────────────────────────────
        # The prompt asks for "a single number" on problems whose stems are
        # denominated in $K or in percent, so an answer of 163195 against a
        # ground truth of 164.7 is a unit reading the prompt never ruled out.
        # Scored as a third accounting rather than folded in silently: it
        # moves the leaderboard, and a reader has to be able to see by how
        # much and on which rows.
        pred_by_id = {p.problem_id: p for p in preds}
        unit_status = [
            ua.classify_prediction(
                pred_by_id[p["id"]].predicted_answer,
                pred_by_id[p["id"]].predicted_interval,
                p,
            )
            for p in sub_problems
        ]
        unit_report = ua.report(unit_status)
        preds_unit, _ = unit_normalized(preds, problems_by_id)
        c_unit, f_unit, o_unit, _ = per_problem_outcomes(sub_problems, preds_unit)
        boot_unit = bootstrap_metrics(c_unit, f_unit, o_unit, conf)
        breakdown_unit = MURUMetrics(sub_problems, preds_unit).compute_all()
        unit_by_category = {}
        for p, raw_ok, unit_ok in zip(sub_problems, c, c_unit):
            cell = unit_by_category.setdefault(
                p["category"], {"n": 0, "raw": 0, "unit_aware": 0}
            )
            cell["n"] += 1
            cell["raw"] += raw_ok
            cell["unit_aware"] += unit_ok

        # ── Hardened calibration metrics (P4) ────────────────────
        hardened = cm.full_summary(c, conf, records=sharpness_records(preds, problems_by_id))
        hardened_unit = cm.full_summary(
            c_unit, conf, records=sharpness_records(preds_unit, problems_by_id)
        )
        overconf = oc.summary(c, conf)
        overconf_unit = oc.summary(c_unit, conf)

        summary[slug] = {
            "display": display,
            "result_file": str(path.relative_to(PROJECT_ROOT)),
            "n_evaluated": len(preds),
            "n_test": len(problems),
            "salvaged": salvaged,
            "subset_difficulty": {
                str(d): v["count"] for d, v in breakdown["accuracy_by_difficulty"].items()
            },
            "metrics": boot,
            "metrics_strict": boot_strict,
            "accounting_gap": {
                "accuracy_pp": 100 * acc_gap,
                "ece": ece_gap,
                "material": abs(acc_gap) > ACCOUNTING_GAP_THRESHOLD,
            },
            "parse": {
                "n_attempted": status["n_attempted"],
                "n_parsed": status["n_parsed"],
                "n_model_failure": status["n_model_failure"],
                "n_provider_failure": status["n_provider_failure"],
                "parse_rate": status["parse_rate"],
                "coverage": status["coverage"],
                "counts": status["counts"],
            },
            "metrics_unit_aware": boot_unit,
            "unit_accounting": {
                **unit_report,
                "by_category": unit_by_category,
            },
            "hardened": hardened,
            "hardened_unit_aware": hardened_unit,
            "overconfidence": overconf,
            "overconfidence_unit_aware": overconf_unit,
            "per_difficulty_acc": {
                str(d): v["accuracy"] for d, v in breakdown["accuracy_by_difficulty"].items()
            },
            "per_category_acc": {
                k: v["accuracy"] for k, v in breakdown["accuracy_by_category"].items()
            },
            "per_difficulty_acc_unit_aware": {
                str(d): v["accuracy"]
                for d, v in breakdown_unit["accuracy_by_difficulty"].items()
            },
            "per_category_acc_unit_aware": {
                k: v["accuracy"]
                for k, v in breakdown_unit["accuracy_by_category"].items()
            },
        }

    # Panel-level cut dependence: whether the comparative overconfidence claim
    # is a property of the models or of the 0.7 threshold. Stored under a
    # reserved underscore key so per-model consumers can skip it.
    panel_oc = oc.panel_cut_dependence(
        {s["display"]: s["overconfidence"]["sweep"] for s in summary.values()}
    )
    summary["_panel_overconfidence"] = panel_oc

    out = PROJECT_ROOT / "evaluation" / "baselines" / "real_llm_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out.relative_to(PROJECT_ROOT)}", file=sys.stderr)

    # Sort entries by accuracy desc; flag partial runs (coverage < 80%) so the
    # paper table can present full-coverage runs first.
    full = []
    partial = []
    for slug, s in summary.items():
        if slug.startswith("_"):
            continue
        s["_slug"] = slug
        if s["n_evaluated"] >= 0.8 * s["n_test"]:
            full.append(s)
        else:
            partial.append(s)
    full.sort(key=lambda s: -s["metrics"]["accuracy"]["point"])
    partial.sort(key=lambda s: -s["metrics"]["accuracy"]["point"])
    ordered = full + partial

    # ─── LaTeX: main results table rows ──────────────────────────
    main_lines = []
    for s in ordered:
        m = s["metrics"]
        n_total = s["n_test"]
        # Coverage now annotates *attempted* problems, not parsed ones. A
        # response the model returned but the parser could not read is a
        # scored event with a status (Table: parse accounting), not missing
        # coverage; only provider-side gaps reduce the denominator here.
        n = s["parse"]["n_attempted"]
        coverage = f"~({n}/{n_total})" if n < n_total else ""
        parse_cell = f"{100 * s['parse']['parse_rate']:.1f}\\%"
        # Rows with no recoverable framework data (fully salvaged) show "n/a"
        # rather than a spurious 0%. Rows with partial framework coverage show
        # the rate over the answered-with-framework subset (n annotated).
        fw = m['framework_match']
        if fw['point'] is None:
            fwmatch_cell = "n/a"
        elif fw.get("n", s["n_evaluated"]) < s["n_evaluated"]:
            fwmatch_cell = fmt_pct(fw['point'], fw['ci95']) + f"$^{{(n={fw['n']})}}$"
        else:
            fwmatch_cell = fmt_pct(fw['point'], fw['ci95'])
        main_lines.append(
            f"{s['display']}{coverage} & "
            f"{parse_cell} & "
            f"{fmt_pct(m['accuracy']['point'], m['accuracy']['ci95'])} & "
            f"{fmt_dec(m['ece']['point'], m['ece']['ci95'])} & "
            f"{fmt_pct(m['overconfidence']['point'], m['overconfidence']['ci95'])} & "
            f"{fwmatch_cell} \\\\"
        )

    # ─── LaTeX: per-difficulty table rows ────────────────────────
    diff_lines = []
    for s in ordered:
        row = [s["display"]]
        for d in ("1", "2", "3", "4", "5"):
            if d in s["per_difficulty_acc"]:
                row.append(f"{100*s['per_difficulty_acc'][d]:.0f}\\%~(n={s['subset_difficulty'][d]})")
            else:
                row.append("---")
        diff_lines.append(" & ".join(row) + " \\\\")

    # ─── LaTeX: per-category table rows ──────────────────────────
    # Columns follow the alphabetical category order used in the paper header:
    # Adversarial Ambiguity, Bayesian Updating, Conditional Chains,
    # Decision Under Uncertainty, Distribution Estimation. Partial-coverage
    # rows are daggered (matching the table caption's footnote).
    cat_order = [
        "adversarial_ambiguity",
        "bayesian_updating",
        "conditional_probability_chains",
        "decision_under_uncertainty",
        "distribution_estimation",
    ]
    cat_lines = []
    for s in ordered:
        dagger = "$^\\dagger$" if s["n_evaluated"] < 0.8 * s["n_test"] else ""
        row = [s["display"] + dagger]
        for cat in cat_order:
            acc = s["per_category_acc"].get(cat)
            row.append(f"{100*acc:.1f}\\%" if acc is not None else "---")
        cat_lines.append(" & ".join(row) + " \\\\")

    # ─── LaTeX: parse-accounting table (P0) ──────────────────────
    # Every headline metric computed twice, so a reader can see exactly how
    # much of each score depends on the choice to drop unreadable responses.
    STATUS_LABEL = {
        "truncated": "trunc.",
        "missing_field": "empty field",
        "format_variant": "off-schema",
        "no_schema": "no schema",
        "refused": "refused",
        "endpoint_unavailable": "endpoint 404",
        "rate_limited": "rate limit",
        "timeout": "timeout",
        "api_error": "API error",
        "empty_response": "empty resp.",
        "unattempted": "unattempted",
    }
    acct_lines = []
    for s in ordered:
        p = s["parse"]
        lenient, strict = s["metrics"], s["metrics_strict"]
        modes = ", ".join(
            f"{n} {STATUS_LABEL.get(k, k)}" for k, n in p["counts"].items() if k != "ok"
        ) or "---"
        gap = s["accounting_gap"]["accuracy_pp"]
        acct_lines.append(
            f"{s['display']} & "
            f"{100 * p['parse_rate']:.1f}\\% & "
            f"{modes} & "
            f"{100 * lenient['accuracy']['point']:.1f}\\% & "
            f"{100 * strict['accuracy']['point']:.1f}\\% & "
            f"{gap:+.1f} & "
            f"{lenient['ece']['point']:.3f} & "
            f"{strict['ece']['point']:.3f} \\\\"
        )

    # ─── LaTeX: sharpness / proper-scoring table (P4) ────────────
    sharp_lines = []
    for s in ordered:
        h = s["hardened"]
        sh = h["sharpness"]
        br = h["brier"]

        def _num(x, fmt="{:.3f}"):
            return fmt.format(x) if x is not None else "---"

        sharp_lines.append(
            f"{s['display']} & "
            f"{_num(sh['pe_coverage'] and 100 * sh['pe_coverage'], '{:.1f}')}\\% & "
            f"{_num(sh['median_relative_width'], '{:.2f}')} & "
            f"{_num(sh['p90_relative_width'], '{:.1f}')} & "
            f"{_num(sh['hedge_rate'] and 100 * sh['hedge_rate'], '{:.1f}')}\\% & "
            f"{_num(sh['median_normalized_interval_score'], '{:.2f}')} & "
            f"{_num(br['brier'])} & "
            f"{_num(br['reliability'])} & "
            f"{_num(br['resolution'])} & "
            f"{_num(h['auroc'])} & "
            f"{_num(h['ece_debiased']['debiased'])} \\\\"
        )

    # ─── LaTeX: unit accounting ──────────────────────────────────
    unit_lines = []
    for s in ordered:
        u = s["unit_accounting"]
        m, mu = s["metrics"], s["metrics_unit_aware"]
        n_err = u["n"] - u["counts"]["correct"]
        unit_lines.append(
            f"{s['display']} & "
            f"{n_err} & "
            f"{u['counts']['unit_mismatch']} & "
            f"{100 * u['unit_mismatch_share_of_errors']:.1f}\\% & "
            f"{100 * m['accuracy']['point']:.1f}\\% & "
            f"{100 * mu['accuracy']['point']:.1f}\\% & "
            f"{100 * (mu['accuracy']['point'] - m['accuracy']['point']):+.1f} & "
            f"{m['ece']['point']:.3f} & "
            f"{mu['ece']['point']:.3f} & "
            f"{100 * m['overconfidence']['point']:.1f}\\% & "
            f"{100 * mu['overconfidence']['point']:.1f}\\% \\\\"
        )

    # ─── LaTeX: overconfidence threshold sensitivity (P4) ────────
    # The rate at five cuts, plus the two statistics that need no cut at all.
    # A threshold sitting on a confidence spike is annotated with the tie mass
    # it straddles, because that is where the strict/inclusive convention
    # stops being a rounding detail.
    SWEEP_TAUS = (0.5, 0.7, 0.8, 0.9, 0.95)
    TIE_MASS_FLAG = 0.10  # annotate a cut this much of the mass sits exactly on
    oc_lines = []
    for s in ordered:
        o = s["overconfidence"]
        by_tau = {r["tau"]: r for r in o["sweep"]}
        cells = []
        for tau in SWEEP_TAUS:
            r = by_tau[tau]
            cell = f"{100 * r['rate_strict']:.1f}\\%"
            if r["tie_mass"] >= TIE_MASS_FLAG:
                cell += f"$^{{\\dagger}}$"
            cells.append(cell)
        cond = o["conditional_error_rate"]
        cge = o["confident_given_error"]
        cond_cell = f"{100 * cond:.1f}\\%" if cond is not None else "---"
        cge_cell = f"{100 * cge:.1f}\\%" if cge is not None else "---"
        oc_lines.append(
            f"{s['display']} & "
            + " & ".join(cells)
            + f" & {o['mean_overconfidence']:.3f}"
            + f" & {cond_cell}"
            + f" & {cge_cell} \\\\"
        )

    # Write to paper-includable .tex files.
    # Trailing %\endinput suppresses the newline LaTeX would otherwise inject
    # at the end of an \input'd file; without it, a stray newline inside the
    # tabular environment triggers "Misplaced \noalign" at the next \bottomrule.
    paper_dir = PROJECT_ROOT / "paper" / "tables"
    paper_dir.mkdir(parents=True, exist_ok=True)
    # Wrap rows in a macro definition. Inside a tabular, \input introduces
    # subtle \par/whitespace issues that trigger "Misplaced \noalign";
    # expanding a macro instead is robust.
    (paper_dir / "real_llm_main.tex").write_text(
        "\\newcommand{\\realllmmainrows}{%\n" + "\n".join(main_lines) + "}\n"
    )
    (paper_dir / "real_llm_difficulty.tex").write_text(
        "\\newcommand{\\realllmdifficultyrows}{%\n" + "\n".join(diff_lines) + "}\n"
    )
    (paper_dir / "real_llm_category.tex").write_text(
        "\\newcommand{\\realllmcategoryrows}{%\n" + "\n".join(cat_lines) + "}\n"
    )
    (paper_dir / "real_llm_accounting.tex").write_text(
        "\\newcommand{\\realllmaccountingrows}{%\n" + "\n".join(acct_lines) + "}\n"
    )
    (paper_dir / "real_llm_sharpness.tex").write_text(
        "\\newcommand{\\realllmsharpnessrows}{%\n" + "\n".join(sharp_lines) + "}\n"
    )
    (paper_dir / "real_llm_overconfidence.tex").write_text(
        "\\newcommand{\\realllmoverconfidencerows}{%\n" + "\n".join(oc_lines) + "}\n"
    )
    (paper_dir / "real_llm_units.tex").write_text(
        "\\newcommand{\\realllmunitrows}{%\n" + "\n".join(unit_lines) + "}\n"
    )
    for name in (
        "real_llm_main",
        "real_llm_difficulty",
        "real_llm_category",
        "real_llm_accounting",
        "real_llm_sharpness",
        "real_llm_overconfidence",
        "real_llm_units",
    ):
        print(f"Saved: paper/tables/{name}.tex", file=sys.stderr)

    # Loud console flag: if any model's two accountings diverge materially,
    # the leaderboard is partly measuring format compliance and the paper has
    # to say so rather than quietly reporting the friendlier number.
    material = [s["display"] for s in ordered if s["accounting_gap"]["material"]]
    if material:
        print(
            f"\n!! Strict/lenient accuracy gap exceeds "
            f"{100 * ACCOUNTING_GAP_THRESHOLD:.0f} pp for: {', '.join(material)}",
            file=sys.stderr,
        )
    else:
        worst = max(ordered, key=lambda s: abs(s["accounting_gap"]["accuracy_pp"]))
        print(
            f"\nParse accounting: max strict/lenient gap "
            f"{abs(worst['accounting_gap']['accuracy_pp']):.1f} pp "
            f"({worst['display']}) — below the {100 * ACCOUNTING_GAP_THRESHOLD:.0f} pp "
            f"threshold, so the leaderboard is not format-compliance-driven.",
            file=sys.stderr,
        )

    # Cut dependence of the comparative overconfidence claim. What has to
    # survive is the ordering and the spread, not the absolute rate — the rate
    # is a function of where the line is drawn and always will be.
    # Unit accounting: how much of the recorded error is a unit reading the
    # prompt never ruled out, and what the leaderboard looks like without it.
    print("\nUnit accounting (corroborated rescales only):", file=sys.stderr)
    for s in ordered:
        u = s["unit_accounting"]
        m, mu = s["metrics"], s["metrics_unit_aware"]
        n_err = u["n"] - u["counts"]["correct"]
        print(
            f"  {s['display']:20s} {n_err:4d} errors, "
            f"{u['counts']['unit_mismatch']:3d} unit mismatches "
            f"({100 * u['unit_mismatch_share_of_errors']:4.1f}% of them)  "
            f"acc {100 * m['accuracy']['point']:5.1f}% -> "
            f"{100 * mu['accuracy']['point']:5.1f}%  "
            f"ECE {m['ece']['point']:.3f} -> {mu['ece']['point']:.3f}",
            file=sys.stderr,
        )
    rank_raw = [s["display"] for s in sorted(ordered, key=lambda s: -s["metrics"]["accuracy"]["point"])]
    rank_unit = [s["display"] for s in sorted(ordered, key=lambda s: -s["metrics_unit_aware"]["accuracy"]["point"])]
    rho_raw = oc.spearman(
        [s["metrics"]["accuracy"]["point"] for s in ordered],
        [s["metrics"]["ece"]["point"] for s in ordered],
    )
    rho_unit = oc.spearman(
        [s["metrics_unit_aware"]["accuracy"]["point"] for s in ordered],
        [s["metrics_unit_aware"]["ece"]["point"] for s in ordered],
    )
    print(
        f"  → accuracy/ECE Spearman: raw {rho_raw:+.2f}, unit-aware {rho_unit:+.2f}"
        f"{'  (ORDERING CHANGES)' if rank_raw != rank_unit else ''}",
        file=sys.stderr,
    )

    print("\nOverconfidence threshold sensitivity:", file=sys.stderr)
    print(
        "  tau    panel range (strict)   spread   ratio   rank-corr vs "
        f"tau={oc.CANONICAL_THRESHOLD}   max tie",
        file=sys.stderr,
    )
    for r in panel_oc["per_threshold"]:
        ratio = f"{r['ratio']:.1f}x" if r["ratio"] else "min=0"
        rc = r["rank_corr_vs_canonical"]
        rci = r["rank_corr_inclusive_vs_canonical"]
        print(
            f"  {r['tau']:.2f}   {100*r['min']:5.1f}% - {100*r['max']:5.1f}%"
            f"      {r['spread_pp']:5.1f} pp  {ratio:>6s}"
            f"   strict {rc:+.2f} / incl {rci:+.2f}"
            f"   {100*r['max_tie_mass']:5.1f}%  n>={r['min_events']:3d}"
            f"{'  [degenerate]' if r['degenerate'] else ''}",
            file=sys.stderr,
        )
    print(
        f"  → over the informative cuts {panel_oc['informative_range']}: "
        f"ordering {'identical to' if panel_oc['orderings_agree'] else 'differs from'} "
        f"tau={oc.CANONICAL_THRESHOLD} "
        f"(min rank-corr {panel_oc['min_rank_corr_informative']:+.2f}); "
        f"weakest/strongest ratio at least {panel_oc['min_ratio_all_cuts']:.1f}x "
        f"across every cut including the degenerate ones.",
        file=sys.stderr,
    )

    # Echo to stdout for convenience.
    print()
    print("% ─── REAL-LLM RESULTS TABLE ───")
    for line in main_lines:
        print(line)
    print()
    print("% ─── REAL-LLM PER-DIFFICULTY TABLE ───")
    for line in diff_lines:
        print(line)


if __name__ == "__main__":
    main()
