#!/usr/bin/env python3
"""
audit_item_defects.py — audit generated problems for item-construction defects.

The judgment pass of 2026-08-17 found defects in the *items*, not the models:
three stems carried physically impossible values, the base-rate template stated
one accuracy figure and quoted another, and some ground-truth intervals were
narrower than the precision the answer format invites. Those were found by
reading 86 sampled model errors, which only surfaces a defect if a model
happened to trip over it. This script checks the whole corpus directly.

Defect classes:

  D1  Physically implausible stem value — the population mean named in a
      sample-mean stem falls outside the admissible range for that quantity.
  D2  Contradictory or ambiguous test accuracy — a base-rate-trap stem whose
      own accuracy figure disagrees with the colleague's quote, or whose
      figure read as overall accuracy implies a sensitivity above 1.
  D3  Ground-truth interval narrower than the invited precision — correct
      arithmetic reported at the natural number of decimals falls outside.

Usage:
    python scripts/audit_item_defects.py                 # audit data/
    python scripts/audit_item_defects.py --split test    # one split
    python scripts/audit_item_defects.py --json          # machine-readable

Exit codes:
    0 — no defects found
    1 — at least one defect found
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS = ("train", "validation", "test")

# ──────────────────────────────────────────────────────────────────────
# D1 — physically admissible ranges for the population mean, by quantity.
# Kept in step with SAMPLE_MEAN_CONTEXTS in scripts/generate_problems.py; the
# bounds here are deliberately looser, so this audit flags only values that are
# implausible on their face rather than merely outside the sampler's band.
# ──────────────────────────────────────────────────────────────────────

PLAUSIBLE_MEAN_RANGE = {
    "response time (ms)": (1, 60_000),
    "weight (grams)": (1, 5_000),
    "battery life (hours)": (0.5, 200),
    "commute time (minutes)": (1, 300),
    "yield per hectare (tonnes)": (0.1, 50),
    "diastolic blood pressure (mmHg)": (30, 130),
    "download speed (Mbps)": (0.1, 10_000),
    "assembly time (seconds)": (1, 10_000),
    "fuel consumption (L/100km)": (1, 60),
    "wait time (minutes)": (1, 1_440),
}

SAMPLE_MEAN_RE = re.compile(
    r"measures the (?P<measurement>[^.]+?) for a sample of \d+ [^.]+?\. "
    r"The sample mean is (?P<mean>[\d.]+)"
)

# ──────────────────────────────────────────────────────────────────────
# D2 — base-rate-trap consistency.
# ──────────────────────────────────────────────────────────────────────

# A stem that says "correctly detects N% of the pedestrians in its path" states
# P(flag | condition) and admits no other reading. A stem that says "detects
# prohibited items with N% accuracy" (the pre-errata wording) does not: read as
# overall accuracy it usually implies a sensitivity above 1, which is what made
# those items unsolvable as written.
TRAP_SENSITIVITY_RE = re.compile(
    r"correctly (?:identifies|detects|flags)\b[^.]*?\b(\d+)% of\b"
)
TRAP_LEGACY_ACC_RE = re.compile(r"\b(?:with|of) (\d+)% accuracy")
TRAP_QUOTE_ACC_RE = re.compile(r"Since the test is (\d+)% accurate")
TRAP_SPEC_RE = re.compile(r"specificity of (\d+)%")
TRAP_BASE_RE = re.compile(r"estimated between ([\d.]+)% and ([\d.]+)%")

# ──────────────────────────────────────────────────────────────────────
# D3 — precision floors.
# A ground-truth interval must be wide enough that an answer rounded to the
# number of decimals the ground truth itself uses can land inside it. We
# require at least two units in the last published decimal place.
# ──────────────────────────────────────────────────────────────────────

MIN_UNITS_IN_LAST_PLACE = 2


def decimals(x: float) -> int:
    """Number of decimal places in the shortest exact repr of x."""
    text = repr(float(x))
    if "e" in text or "E" in text:
        return 12
    return len(text.partition(".")[2].rstrip("0"))


def load_problems(splits) -> list[dict]:
    problems = []
    for split in splits:
        directory = DATA_DIR / split
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("MURU-*.json")):
            with open(path) as handle:
                problem = json.load(handle)
            problem["_split"] = split
            problem["_path"] = str(path.relative_to(PROJECT_ROOT))
            problems.append(problem)
    return problems


def check_d1(problem: dict) -> list[str]:
    """Physically implausible stem value."""
    if problem["metadata"].get("author") != "generator_sample_mean":
        return []
    match = SAMPLE_MEAN_RE.search(problem["stem"])
    if not match:
        return []
    measurement = match.group("measurement")
    mean = float(match.group("mean"))
    bounds = PLAUSIBLE_MEAN_RANGE.get(measurement)
    if bounds is None:
        return [f"D1: no plausible range registered for measurement '{measurement}'"]
    low, high = bounds
    if not (low <= mean <= high):
        return [
            f"D1: sample mean {mean} is outside the plausible range "
            f"[{low}, {high}] for '{measurement}'"
        ]
    return []


def check_d2(problem: dict) -> list[str]:
    """Contradictory or ambiguous test accuracy in a base-rate trap."""
    if problem["metadata"].get("author") != "generator_base_rate_trap":
        return []
    stem = problem["stem"]
    errors = []

    explicit = TRAP_SENSITIVITY_RE.search(stem)
    legacy = TRAP_LEGACY_ACC_RE.search(stem)
    quote_acc = TRAP_QUOTE_ACC_RE.search(stem)
    spec = TRAP_SPEC_RE.search(stem)
    base = TRAP_BASE_RE.search(stem)
    if not (quote_acc and spec and base) or not (explicit or legacy):
        return ["D2: stem does not match the expected base-rate-trap shape"]

    stated = int((explicit or legacy).group(1))
    quoted = int(quote_acc.group(1))
    if stated != quoted:
        errors.append(
            f"D2: stem states {stated}% but the colleague's quote uses {quoted}%"
        )

    # Only a stem that leaves the figure as bare "accuracy" can be read as
    # overall accuracy: acc = sens·p + spec·(1-p). Where that reading implies a
    # sensitivity above 1 the item is unsolvable as written — Qwen3-32B derived
    # 3.29, correctly rejected it, and never answered.
    if explicit is None:
        specificity = int(spec.group(1)) / 100
        prevalence = (float(base.group(1)) + float(base.group(2))) / 200
        if prevalence > 0:
            implied = (stated / 100 - specificity * (1 - prevalence)) / prevalence
            if implied > 1:
                errors.append(
                    f"D2: the figure is stated as bare accuracy; read as overall "
                    f"accuracy it implies sensitivity {implied:.2f} > 1 (accuracy "
                    f"{stated}%, specificity {specificity:.2f}, prevalence "
                    f"{prevalence:.4f}) — unsolvable under that reading"
                )
    return errors


def check_d3(problem: dict) -> list[str]:
    """Ground-truth interval narrower than the precision it invites."""
    ci = problem["ground_truth"].get("confidence_interval")
    if not ci:
        return []
    low, high = ci
    width = high - low
    places = max(decimals(low), decimals(high), decimals(problem["ground_truth"]["point_estimate"]))
    floor = MIN_UNITS_IN_LAST_PLACE * (10 ** -places)
    if width < floor - 1e-12:
        return [
            f"D3: ground-truth interval [{low}, {high}] is {width:.5g} wide, under "
            f"{MIN_UNITS_IN_LAST_PLACE} units of the last published decimal "
            f"({floor:.5g}) — correct arithmetic can fail on rounding alone"
        ]
    return []


CHECKS = (check_d1, check_d2, check_d3)


def audit(problems: list[dict]) -> list[dict]:
    findings = []
    for problem in problems:
        for check in CHECKS:
            for message in check(problem):
                findings.append(
                    {
                        "id": problem["id"],
                        "split": problem["_split"],
                        "path": problem["_path"],
                        "template": problem["metadata"].get("author"),
                        "defect": message.split(":", 1)[0],
                        "message": message,
                    }
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--split", choices=SPLITS, help="Audit a single split.")
    parser.add_argument("--json", action="store_true", help="Emit findings as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Print the summary only.")
    args = parser.parse_args()

    splits = (args.split,) if args.split else SPLITS
    problems = load_problems(splits)
    findings = audit(problems)

    if args.json:
        json.dump(
            {"n_problems": len(problems), "n_findings": len(findings), "findings": findings},
            sys.stdout,
            indent=2,
        )
        print()
        return 1 if findings else 0

    print(f"\n  Audited {len(problems)} problems across {', '.join(splits)}\n")
    if not findings:
        print("  No item-construction defects found.\n")
        return 0

    by_defect = Counter(f["defect"] for f in findings)
    by_split = Counter(f["split"] for f in findings)
    affected = {f["id"] for f in findings}

    if not args.quiet:
        for finding in findings:
            print(f"  {finding['split']:<10} {finding['id']}  {finding['message']}")
        print()

    print(f"  {len(findings)} findings over {len(affected)} distinct problems")
    print("  by class: " + ", ".join(f"{k}={v}" for k, v in sorted(by_defect.items())))
    print("  by split: " + ", ".join(f"{k}={v}" for k, v in sorted(by_split.items())))
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
