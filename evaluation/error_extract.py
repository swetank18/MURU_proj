#!/usr/bin/env python3
"""
error_extract.py — Pull every wrong answer out of the committed archives, with
the problem it got wrong, so the failures can be read rather than counted.

The leaderboard says how often each model is wrong. It cannot say what the
model did instead, and "models are miscalibrated on Decision-Under-Uncertainty"
is a number, not a diagnosis. This module assembles the corpus the failure
taxonomy is coded against: one record per incorrect prediction, carrying the
model's full response text next to the ground-truth derivation, so a coder
(human or LLM judge) sees exactly what the model saw and what it should have
produced.

Two things it deliberately does not do:

  * It does not decide what the failure modes are. The codebook is written
    after reading the sample, not before, because a taxonomy invented from the
    armchair will find exactly the categories it invented.
  * It does not silently drop uncodeable records. Answers recovered from
    progress logs during multi-day accumulation kept their parsed fields but
    lost the raw response text, so they can be counted but not read. They are
    emitted with ``codeable: false`` and excluded from the sample, and the
    counts are reported, because a coding sample drawn only from the readable
    subset is a biased sample unless you can see how much was set aside.

Outputs (under evaluation/errors/):
  errors.jsonl          every incorrect prediction in the panel, one per line
  sample_<n>.json       a seeded stratified sample for hand-coding
  summary.json          per-model error counts and codeable fractions

Usage:
  python evaluation/error_extract.py                # full corpus + n=100 sample
  python evaluation/error_extract.py --sample 60 --seed 7
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.aggregate_real_llm import DISPLAY_NAMES, find_latest_result, load_problems
from evaluation.parse_status import MODEL_FAILURE_STATUSES, classify_entry

OUT_DIR = PROJECT_ROOT / "evaluation" / "errors"

# Marker written into the response field by the salvage path when the raw text
# could not be recovered from a progress log.
SALVAGE_MARKER = "[salvaged from log"


def is_codeable(response):
    """Can a coder actually read what the model did?"""
    return bool(response) and not response.startswith(SALVAGE_MARKER)


def error_records(slug, display, data, problems_by_id):
    """Every incorrect or unreadable prediction from one model's archive."""
    out = []
    for r in data.get("raw_results", []):
        pid = r.get("problem_id")
        problem = problems_by_id.get(pid)
        if problem is None:
            continue
        parsed = r.get("parsed") or {}
        pe = parsed.get("point_estimate")
        gt = problem["ground_truth"]
        lo, hi = gt["confidence_interval"]

        status = classify_entry(r)
        parse_failure = status in MODEL_FAILURE_STATUSES
        if pe is None:
            # No readable answer. A model-side failure is a failure of the
            # same reasoning process the benchmark measures and belongs in the
            # corpus; a provider-side one is missing data and does not.
            if not parse_failure:
                continue
            correct = False
        else:
            correct = lo <= pe <= hi
            if correct:
                continue

        response = r.get("response", "")
        out.append({
            "model": display,
            "model_slug": slug,
            "problem_id": pid,
            "category": problem["category"],
            "difficulty": problem["difficulty"],
            "required_framework": problem["required_framework"],
            "uncertainty_type": problem.get("uncertainty_type"),
            "stem": problem["stem"],
            "gt_point": gt["point_estimate"],
            "gt_ci": gt["confidence_interval"],
            "gt_answer": gt.get("answer"),
            "solution_steps": problem.get("solution_steps", []),
            "authored_failure_modes": problem.get("common_failure_modes", []),
            "model_point_estimate": pe,
            "model_ci": parsed.get("confidence_interval"),
            "model_confidence": parsed.get("confidence"),
            "model_framework": parsed.get("framework"),
            "framework_match": parsed.get("framework") == problem["required_framework"],
            "parse_status": status,
            "parse_failure": parse_failure,
            "response": response,
            "codeable": is_codeable(response),
        })
    return out


def stratified_sample(records, n, seed):
    """Seeded sample stratified by (model, category).

    Proportional allocation with a floor of one per non-empty cell and
    largest-remainder rounding, so the panel leader's 22 errors are not
    swamped by the weakest model's 166 and no category drops out entirely.
    Only codeable records are eligible.
    """
    pool = [r for r in records if r["codeable"]]
    if n >= len(pool):
        return sorted(pool, key=lambda r: (r["model"], r["problem_id"]))

    cells = defaultdict(list)
    for r in pool:
        cells[(r["model"], r["category"])].append(r)

    total = len(pool)
    exact = {k: n * len(v) / total for k, v in cells.items()}
    alloc = {k: min(len(cells[k]), max(1, int(v))) for k, v in exact.items()}

    # Largest-remainder pass to land exactly on n without exceeding any cell.
    while sum(alloc.values()) != n:
        short = n - sum(alloc.values())
        if short > 0:
            room = [k for k in cells if alloc[k] < len(cells[k])]
            if not room:
                break
            room.sort(key=lambda k: exact[k] - alloc[k], reverse=True)
            for k in room[:short]:
                alloc[k] += 1
        else:
            over = [k for k in cells if alloc[k] > 1]
            if not over:
                break
            over.sort(key=lambda k: alloc[k] - exact[k], reverse=True)
            for k in over[: -short]:
                alloc[k] -= 1

    rng = random.Random(seed)
    picked = []
    for key in sorted(cells):
        bucket = sorted(cells[key], key=lambda r: r["problem_id"])
        rng.shuffle(bucket)
        picked.extend(bucket[: alloc[key]])
    return sorted(picked, key=lambda r: (r["model"], r["problem_id"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100, help="hand-coding sample size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    problems = load_problems(PROJECT_ROOT / "data" / "test")
    problems_by_id = {p["id"]: p for p in problems}

    records = []
    for slug, display in DISPLAY_NAMES.items():
        path = find_latest_result(slug)
        if not path:
            continue
        with open(path) as f:
            data = json.load(f)
        if not data.get("raw_results"):
            continue
        records.extend(error_records(slug, display, data, problems_by_id))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda r: (r["model"], r["problem_id"]))
    with open(OUT_DIR / "errors.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    sample = stratified_sample(records, args.sample, args.seed)
    sample_path = OUT_DIR / f"sample_{len(sample)}.json"
    with open(sample_path, "w") as f:
        json.dump({"seed": args.seed, "n": len(sample), "records": sample}, f, indent=2)

    by_model = defaultdict(lambda: {"errors": 0, "codeable": 0, "parse_failures": 0})
    for r in records:
        m = by_model[r["model"]]
        m["errors"] += 1
        m["codeable"] += int(r["codeable"])
        m["parse_failures"] += int(r["parse_failure"])
    summary = {
        "n_errors": len(records),
        "n_codeable": sum(1 for r in records if r["codeable"]),
        "by_model": dict(by_model),
        "by_category": dict(Counter(r["category"] for r in records)),
        "by_difficulty": dict(Counter(r["difficulty"] for r in records)),
        "sample": {"file": sample_path.name, "n": len(sample), "seed": args.seed},
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{len(records)} errors across {len(by_model)} models "
          f"({summary['n_codeable']} with readable responses)")
    for model, m in sorted(by_model.items(), key=lambda kv: -kv[1]["errors"]):
        print(f"  {model:20s} {m['errors']:4d} errors  "
              f"{m['codeable']:4d} codeable  {m['parse_failures']:3d} unreadable answers")
    print(f"\nWrote {OUT_DIR.relative_to(PROJECT_ROOT)}/errors.jsonl, "
          f"{sample_path.name} (n={len(sample)}, seed={args.seed}), summary.json")


if __name__ == "__main__":
    main()
