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
from evaluation import unit_accounting as ua

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
            # A corroborated unit mismatch is a right answer in a different
            # admissible unit, not a reasoning failure. It stays in the corpus
            # (the count is a finding in its own right) but is excluded from
            # the coding sample, which is for reading actual failures.
            "unit_status": ua.classify_prediction(
                pe, parsed.get("confidence_interval"), problem
            ),
            "response": response,
            "codeable": is_codeable(response),
        })
    return out


def stratified_sample(records, n, seed, balanced=True, cap=25):
    """Seeded coding sample: balanced across models by default.

    Eligible records are those with a readable response that are not
    corroborated unit mismatches --- the latter are right answers in another
    admissible unit (``unit_accounting``), and coding them as reasoning
    failures would be coding our own prompt.

    Two allocations, because they answer different questions:

      * **balanced** (default) --- up to ``cap`` errors per model, stratified
        by category within each model. The taxonomy's central question is
        whether failure modes *shift* with capability or merely shrink, and
        that is a within-model composition question. Proportional allocation
        cannot answer it here: the weakest model contributes half the panel's
        errors, so a proportional sample of 100 is two-thirds one model and
        the strongest models arrive with a handful of items each.
      * **proportional** --- allocation by error mass, which is the right
        design for "what does the panel's total error consist of" and the
        wrong one for the question above.

    Neither can manufacture items that do not exist: GPT-OSS-120B has nine
    codeable errors in total, so its cell is nine however the sample is drawn,
    and any per-model claim about it is bounded by that.
    """
    pool = [
        r for r in records
        if r["codeable"] and r["unit_status"] != "unit_mismatch"
    ]
    rng = random.Random(seed)

    if balanced:
        picked = []
        for model in sorted({r["model"] for r in pool}):
            rows = [r for r in pool if r["model"] == model]
            picked.extend(_draw_within(rows, min(cap, len(rows)), rng))
        return sorted(picked, key=lambda r: (r["model"], r["problem_id"]))

    if n >= len(pool):
        return sorted(pool, key=lambda r: (r["model"], r["problem_id"]))
    return sorted(
        _draw_within(pool, n, rng, key=lambda r: (r["model"], r["category"])),
        key=lambda r: (r["model"], r["problem_id"]),
    )


def _draw_within(rows, n, rng, key=None):
    """Proportional-with-floor draw over strata, largest remainder to hit n."""
    if n >= len(rows):
        return list(rows)
    key = key or (lambda r: r["category"])
    cells = defaultdict(list)
    for r in rows:
        cells[key(r)].append(r)

    total = len(rows)
    exact = {k: n * len(v) / total for k, v in cells.items()}
    alloc = {k: min(len(cells[k]), max(1, int(v))) for k, v in exact.items()}
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
            for k in over[:-short]:
                alloc[k] -= 1

    picked = []
    for k in sorted(cells, key=str):
        bucket = sorted(cells[k], key=lambda r: r["problem_id"])
        rng.shuffle(bucket)
        picked.extend(bucket[: alloc[k]])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100, help="sample size (proportional mode only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=25, help="max coded errors per model in balanced mode")
    ap.add_argument("--proportional", action="store_true", help="allocate by error mass instead of balancing across models")
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

    sample = stratified_sample(
        records, args.sample, args.seed,
        balanced=not args.proportional, cap=args.cap,
    )
    sample_path = OUT_DIR / f"sample_{len(sample)}.json"
    with open(sample_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "n": len(sample),
            "allocation": "proportional" if args.proportional else f"balanced (cap {args.cap}/model)",
            "records": sample,
        }, f, indent=2)

    by_model = defaultdict(lambda: {"errors": 0, "codeable": 0, "parse_failures": 0})
    for r in records:
        m = by_model[r["model"]]
        m["errors"] += 1
        m["codeable"] += int(r["codeable"])
        m["parse_failures"] += int(r["parse_failure"])
    summary = {
        "n_errors": len(records),
        "n_codeable": sum(1 for r in records if r["codeable"]),
        "n_unit_mismatch": sum(
            1 for r in records if r["unit_status"] == "unit_mismatch"
        ),
        "by_model": dict(by_model),
        "by_category": dict(Counter(r["category"] for r in records)),
        "by_difficulty": dict(Counter(r["difficulty"] for r in records)),
        "sample": {"file": sample_path.name, "n": len(sample), "seed": args.seed},
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{len(records)} errors across {len(by_model)} models "
          f"({summary['n_codeable']} with readable responses, "
          f"{summary['n_unit_mismatch']} of them right answers in another unit)")
    for model, m in sorted(by_model.items(), key=lambda kv: -kv[1]["errors"]):
        print(f"  {model:20s} {m['errors']:4d} errors  "
              f"{m['codeable']:4d} codeable  {m['parse_failures']:3d} unreadable answers")
    print(f"\nWrote {OUT_DIR.relative_to(PROJECT_ROOT)}/errors.jsonl, "
          f"{sample_path.name} (n={len(sample)}, seed={args.seed}), summary.json")


if __name__ == "__main__":
    main()
