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


N_BOOTSTRAP = 10_000

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

        # Difficulty / category breakdown via canonical metrics
        m = MURUMetrics(sub_problems, preds)
        breakdown = m.compute_all()

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
            "per_difficulty_acc": {
                str(d): v["accuracy"] for d, v in breakdown["accuracy_by_difficulty"].items()
            },
            "per_category_acc": {
                k: v["accuracy"] for k, v in breakdown["accuracy_by_category"].items()
            },
        }

    out = PROJECT_ROOT / "evaluation" / "baselines" / "real_llm_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out.relative_to(PROJECT_ROOT)}", file=sys.stderr)

    # Sort entries by accuracy desc; flag partial runs (coverage < 80%) so the
    # paper table can present full-coverage runs first.
    full = []
    partial = []
    for slug, s in summary.items():
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
        n = s["n_evaluated"]
        n_total = s["n_test"]
        coverage = f"~({n}/{n_total})" if n < n_total else ""
        # Salvaged rows have no framework data — display as "n/a" rather than
        # let the table read 0%.
        fwmatch_cell = (
            "n/a"
            if s.get("salvaged")
            else fmt_pct(m['framework_match']['point'], m['framework_match']['ci95'])
        )
        main_lines.append(
            f"{s['display']}{coverage} & "
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
    print(f"Saved: paper/tables/real_llm_main.tex", file=sys.stderr)
    print(f"Saved: paper/tables/real_llm_difficulty.tex", file=sys.stderr)

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
