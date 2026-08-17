#!/usr/bin/env python3
"""
failure_codebook.py — What the panel's wrong answers actually are.

The codebook below was written after reading errors, not before. That ordering
matters: a taxonomy invented from the armchair finds exactly the categories it
invented, and the two largest things in this one --- answers that are right but
reported in the wrong unit, and answers that report a *different quantity* from
the same computation --- were not on the list we would have written first.

Codes fall into three groups, and only one of them is about arithmetic:

  REPORTING (the computation is right, the reported number is not the answer)
    R1  wrong-quantity        reports an input or an intermediate instead of
                              the asked-for output --- e.g. on value-of-
                              information problems, the *price* of the
                              information rather than its value
    R2  ambiguous-target      reports the other admissible answer on a stem
                              that does not say which one it wants
    R3  wrong-summary         reports an endpoint of its own stated interval
                              instead of a central estimate
    R4  unit-mismatch         right number, different admissible unit
                              (scored separately: ``unit_accounting``)

  COMPUTATION
    C1  direction             right magnitude, wrong sign or wrong direction
                              of a correction
    C2  uncertainty-collapse  treats an uncertain input as fixed, producing a
                              point-like interval
    C3  arithmetic            wrong formula or slip inside the right framework
    C4  single-formalisation  commits to one reading of a deliberately
                              ambiguous problem
    C5  transcription         mis-copies a stated quantity

  REPORTING OF CONFIDENCE
    M1  acknowledged-error    the reasoning states the calculation is wrong,
                              rough, or simplified; the stated confidence does
                              not reflect that
    M2  false-precision       zero-width interval, usually with confidence 1.0

  FORMAT
    S1  off-schema            answer present but not in the requested block

Four more codes were added after the judgment pass (``judgment_coding.py``) read
the sample. They are listed separately because they were *not* derivable from
the mechanical pass, and because two of them are not failures of the model at
all --- reading the errors turned up defects in the benchmark:

  R5  answer-derivation-mismatch
                              the schema block reports a number that does not
                              appear in, and does not follow from, the model's
                              own completed derivation
  A1  asserted-execution      presents code or a numerical procedure it never
                              ran and reports a fabricated result. Distinct
                              from having no derivation: the method is often
                              correct, and executing it would answer the item
  T1  tolerance-artefact      *item defect.* The answer is right to the
                              precision the stem invites, and the ground-truth
                              interval is narrower than that precision
  D1  defective-item          *item defect.* The stem is internally
                              inconsistent or physically impossible, and the
                              model's response is a reasonable reaction to that

Detection is deliberately split. R2, R3, R4, M2 and S1 are decidable from the
record alone and are detected here, with the rule for each written down. C1--C5
and M1 require reading the reasoning and are left to the human/judge pass ---
``code_mechanical`` returns ``None`` for them rather than guessing, so the
model x failure-mode matrix this module produces is honest about which cells it
can fill and which are still open.

One cross-cutting measurement needs no codebook at all and is reported
alongside: how often a wrong answer's *own stated interval* contains the ground
truth. That is the four-field schema earning its keep --- a benchmark scoring
only a point estimate cannot see the difference between a model that does not
know and a model that knows and reports the wrong summary.

Usage:
  python evaluation/error_extract.py      # build evaluation/errors/errors.jsonl
  python evaluation/failure_codebook.py   # code it, print the matrix, emit LaTeX
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Codes that a rule can decide from the record alone.
MECHANICAL_CODES = ("R2", "R3", "R4", "M2", "S1")

# Codes that require reading the reasoning chain.
JUDGMENT_CODES = ("R1", "R5", "C1", "C2", "C3", "C4", "C5", "M1", "A1")

# Codes that describe a defect in the item rather than in the model. Kept
# separate everywhere: counting these as model failures would inflate the
# error rate with our own construction bugs.
ITEM_DEFECT_CODES = ("T1", "D1")

CODE_LABELS = {
    "R1": "wrong-quantity",
    "R2": "ambiguous-target",
    "R3": "wrong-summary",
    "R4": "unit-mismatch",
    "C1": "direction",
    "C2": "uncertainty-collapse",
    "C3": "arithmetic",
    "C4": "single-formalisation",
    "C5": "transcription",
    "M1": "acknowledged-error",
    "M2": "false-precision",
    "S1": "off-schema",
    "R5": "answer-derivation-mismatch",
    "A1": "asserted-execution",
    "T1": "tolerance-artefact",
    "D1": "defective-item",
}

# R3: how close to an endpoint of its own interval the point estimate has to be
# before "it reported the edge of its range" is the better description than "it
# reported a central estimate that happens to sit near the edge".
ENDPOINT_TOLERANCE = 0.02

# R2: the two-option expected-value template. The stem gives one option a
# stated guaranteed return and asks for "the expected value of each option";
# the schema accepts one number, and the ground truth silently takes the
# uncertain option's. A model answering the guaranteed value has read the
# question the other admissible way -- and has done no arithmetic to do it,
# which is why this is coded as a defect of the item rather than credited.
_GUARANTEED = re.compile(r"guaranteed return of \$?([\d.]+)\s*K", re.IGNORECASE)


def guaranteed_value(stem):
    """The stated risk-free payoff on the two-option template, if present."""
    m = _GUARANTEED.search(stem or "")
    return float(m.group(1)) if m else None


def _close(a, b, rel=0.005):
    return b is not None and a is not None and abs(a - b) <= rel * max(abs(b), 1.0)


def code_mechanical(record):
    """Codes decidable from the record. Returns a list, possibly empty.

    ``record`` is one entry of ``evaluation/errors/errors.jsonl``.
    """
    codes = []
    pe = record.get("model_point_estimate")
    ci = record.get("model_ci")
    gt_lo, gt_hi = record.get("gt_ci", (None, None))

    if record.get("parse_status") in ("format_variant", "no_schema"):
        codes.append("S1")

    if record.get("unit_status") == "unit_mismatch":
        codes.append("R4")

    if pe is not None:
        guaranteed = guaranteed_value(record.get("stem"))
        if guaranteed is not None and any(
            _close(pe / k, guaranteed) for k in (1.0, 1000.0)
        ):
            codes.append("R2")

    if ci and len(ci) == 2 and all(x is not None for x in ci):
        lo, hi = sorted(ci)
        if lo == hi:
            codes.append("M2")
        elif pe is not None:
            width = hi - lo
            if width > 0 and (
                abs(pe - lo) <= ENDPOINT_TOLERANCE * width
                or abs(pe - hi) <= ENDPOINT_TOLERANCE * width
            ):
                codes.append("R3")

    return codes


def interval_covers_truth(record):
    """Did the model's own interval contain the ground-truth estimate?

    Only meaningful on records already scored incorrect: it isolates the
    answers where the model's uncertainty representation was right and its
    point summary was not.
    """
    ci = record.get("model_ci")
    gt = record.get("gt_point")
    if not ci or len(ci) != 2 or any(x is None for x in ci) or gt is None:
        return None
    lo, hi = sorted(ci)
    if lo == hi:
        return None
    return lo <= gt <= hi


def matrix(records):
    """Model x code counts over the mechanically-decidable codes.

    Returns ``{model: {code: n, ...,
                       "_errors": n, "_interval_covers_truth": n,
                       "_uncoded": n}}``. ``_uncoded`` counts errors carrying no
    mechanical code at all --- the share of the corpus that the judgment pass
    still has to account for, and the honest denominator for any claim that
    this taxonomy explains the panel's failures.
    """
    out = {}
    for r in records:
        cell = out.setdefault(
            r["model"],
            {c: 0 for c in MECHANICAL_CODES}
            | {"_errors": 0, "_interval_covers_truth": 0, "_uncoded": 0},
        )
        cell["_errors"] += 1
        codes = code_mechanical(r)
        for c in codes:
            cell[c] += 1
        if not codes:
            cell["_uncoded"] += 1
        if interval_covers_truth(r):
            cell["_interval_covers_truth"] += 1
    return out


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

ERRORS = PROJECT_ROOT / "evaluation" / "errors" / "errors.jsonl"

# Display order: strongest model first, matching the leaderboard.
MODEL_ORDER = (
    "GPT-OSS-120B",
    "Qwen3-32B",
    "Llama-4-Scout-17B",
    "Llama-3.3-70B",
    "Llama-3.1-8B",
)


def load_errors(path=ERRORS):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ordered_models(counts):
    known = [m for m in MODEL_ORDER if m in counts]
    return known + sorted(m for m in counts if m not in MODEL_ORDER)


def split_corpus(records):
    """Raw errors, corroborated unit mismatches, and what is left.

    The corpus is built against the raw scoring, so a quarter of it is answers
    the unit accounting credits as correct. Those are reported as their own
    column (R4) and excluded from every other count: coding them as reasoning
    failures would be coding our own prompt, and leaving them in would let the
    taxonomy claim explanatory coverage it has not earned.
    """
    mismatches = [r for r in records if r.get("unit_status") == "unit_mismatch"]
    remaining = [r for r in records if r.get("unit_status") != "unit_mismatch"]
    return mismatches, remaining


def rows(records):
    """Per-model row data for both the console table and the paper table."""
    by_model_all = matrix(records)
    mismatches, remaining = split_corpus(records)
    by_model_rem = matrix(remaining)
    mm_counts = matrix(mismatches)

    out = []
    for model in ordered_models(by_model_all):
        rem = by_model_rem.get(
            model,
            {c: 0 for c in MECHANICAL_CODES}
            | {"_errors": 0, "_interval_covers_truth": 0, "_uncoded": 0},
        )
        out.append({
            "model": model,
            "raw_errors": by_model_all[model]["_errors"],
            "unit_mismatches": mm_counts.get(model, {}).get("_errors", 0),
            "remaining": rem["_errors"],
            "codes": {c: rem[c] for c in MECHANICAL_CODES if c != "R4"},
            "interval_covers_truth": rem["_interval_covers_truth"],
            "uncoded": rem["_uncoded"],
        })
    return out


def _pct(k, n):
    return f"{100 * k / n:.0f}" if n else "---"


def write_latex(row_data, path=None):
    """Emit the paper table, same macro convention as the other generators."""
    codes = [c for c in MECHANICAL_CODES if c != "R4"]
    lines = []
    for r in row_data:
        n = r["remaining"]
        cells = " & ".join(
            f"{r['codes'][c]}" if r["codes"][c] else "---" for c in codes
        )
        lines.append(
            f"{r['model']} & {r['raw_errors']} & {r['unit_mismatches']} & {n} & "
            f"{cells} & "
            f"{r['interval_covers_truth']} ({_pct(r['interval_covers_truth'], n)}\\%) & "
            f"{r['uncoded']} ({_pct(r['uncoded'], n)}\\%) \\\\"
        )
    path = path or PROJECT_ROOT / "paper" / "tables" / "failure_codes.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\\newcommand{\\failurecoderows}{%\n" + "\n".join(lines) + "}\n"
    )
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", default=str(ERRORS))
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    path = Path(args.errors)
    if not path.exists():
        print(f"No corpus at {path}. Run evaluation/error_extract.py first.",
              file=sys.stderr)
        return 1

    records = load_errors(path)
    row_data = rows(records)
    codes = [c for c in MECHANICAL_CODES if c != "R4"]

    header = (f"{'model':20s} {'raw':>5s} {'R4':>4s} {'rem':>5s} "
              + " ".join(f"{c:>4s}" for c in codes)
              + f" {'CI covers':>11s} {'uncoded':>11s}")
    print(f"Failure codes over {len(records)} raw errors. R4 (corroborated unit "
          f"mismatch) is\ncredited correct by the unit accounting and excluded "
          f"from every later column.\n")
    print(header)
    print("-" * len(header))
    for r in row_data:
        n = r["remaining"]
        print(f"{r['model']:20s} {r['raw_errors']:5d} {r['unit_mismatches']:4d} {n:5d} "
              + " ".join(f"{r['codes'][c]:4d}" for c in codes)
              + f" {r['interval_covers_truth']:5d} ({_pct(r['interval_covers_truth'], n):>3s}%)"
              + f" {r['uncoded']:5d} ({_pct(r['uncoded'], n):>3s}%)")

    raw_all = sum(r["raw_errors"] for r in row_data)
    mm_all = sum(r["unit_mismatches"] for r in row_data)
    rem_all = sum(r["remaining"] for r in row_data)
    total = {c: sum(r["codes"][c] for r in row_data) for c in codes}
    uncoded = sum(r["uncoded"] for r in row_data)
    covers = sum(r["interval_covers_truth"] for r in row_data)
    print("-" * len(header))
    print(f"{'panel':20s} {raw_all:5d} {mm_all:4d} {rem_all:5d} "
          + " ".join(f"{total[c]:4d}" for c in codes)
          + f" {covers:5d} ({_pct(covers, rem_all):>3s}%)"
          + f" {uncoded:5d} ({_pct(uncoded, rem_all):>3s}%)")

    print("\nCodes: " + ", ".join(f"{c} {CODE_LABELS[c]}" for c in MECHANICAL_CODES))
    print(f"\n{uncoded} of the {rem_all} errors that survive the unit correction "
          f"({100 * uncoded / rem_all:.0f}%) carry no mechanically-decidable\n"
          f"code. That is the denominator the {', '.join(JUDGMENT_CODES)} "
          f"judgment pass has to account for.")
    print(f"{covers} ({100 * covers / rem_all:.0f}%) are wrong point estimates "
          f"whose own stated interval contains\nthe ground truth --- invisible "
          f"to a benchmark that scores only a point estimate.")

    out = path.parent / "failure_codes.json"
    with open(out, "w") as f:
        json.dump({
            "n_raw_errors": raw_all,
            "n_unit_mismatches": mm_all,
            "n_remaining": rem_all,
            "mechanical_codes": list(MECHANICAL_CODES),
            "judgment_codes": list(JUDGMENT_CODES),
            "by_model": row_data,
            "panel": total | {
                "raw_errors": raw_all,
                "unit_mismatches": mm_all,
                "remaining": rem_all,
                "uncoded": uncoded,
                "interval_covers_truth": covers,
            },
        }, f, indent=2)
    print(f"\nSaved: {out.relative_to(PROJECT_ROOT)}")
    if not args.no_latex:
        print(f"Saved: {write_latex(row_data).relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
