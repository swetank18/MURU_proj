#!/usr/bin/env python3
"""
robustness_subset.py — Fix the 100-problem subset every ablation runs on.

The robustness experiments (repeated sampling, prompt-format variants, a
re-seeded contamination probe, a clean-prompt replication) each multiply the
number of API calls by the number of arms. Running them on the full 301-problem
test split is affordable for none of them on a free tier, and running each one
on a *different* convenience sample makes the arms incomparable. So the subset
is drawn once, committed, and reused: differences between arms are then within
-problem differences, and a paired test applies.

Two constraints shape the draw:

  * **Stratified by category and difficulty.** An unstratified sample of 100
    from 301 leaves the D5 band with single digits and can miss a category
    almost entirely, which is exactly where the ablations are expected to bite.
  * **Restricted to problems every live endpoint already answered.** The
    ablation arms are compared against the archived panel, so an item only
    earns its place if the baseline exists for every model that will be re-run.
    Two of the original five endpoints have been withdrawn by the provider and
    are excluded from that intersection rather than shrinking it.

Output: data/robustness_subset.json --- the ID list plus the provenance needed
to redraw it (seed, stratification, eligibility rule, per-cell counts).

Usage:
  python scripts/robustness_subset.py            # write the subset
  python scripts/robustness_subset.py --check    # verify the committed file
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.aggregate_real_llm import find_latest_result, load_problems

OUT = PROJECT_ROOT / "data" / "robustness_subset.json"

# Endpoints still served by the provider, and therefore re-runnable. Qwen3-32B
# and Llama-4-Scout-17B are omitted deliberately: they were withdrawn, so no
# ablation arm can ever include them and requiring their coverage would only
# shrink the eligible pool.
LIVE_SLUGS = ("gpt-oss-120b", "llama-3_3-70b", "llama-3_1-8b")

SUBSET_N = 100
SEED = 20260816


def answered_ids(slug):
    """Problem IDs this model answered with a readable point estimate."""
    path = find_latest_result(slug)
    if not path:
        return set()
    with open(path) as f:
        data = json.load(f)
    return {
        r["problem_id"]
        for r in data.get("raw_results", [])
        if r.get("success") and (r.get("parsed") or {}).get("point_estimate") is not None
    }


def stratified_draw(problems, n, seed):
    """Proportional allocation over (category, difficulty) with a floor of one.

    Largest-remainder rounding lands exactly on ``n`` without over-drawing any
    cell. Within a cell the pick is a seeded shuffle of the ID-sorted list, so
    the draw is reproducible from the committed seed alone.
    """
    cells = defaultdict(list)
    for p in problems:
        cells[(p["category"], p["difficulty"])].append(p)

    total = len(problems)
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

    rng = random.Random(seed)
    picked = []
    for key in sorted(cells):
        bucket = sorted(cells[key], key=lambda p: p["id"])
        rng.shuffle(bucket)
        picked.extend(bucket[: alloc[key]])
    return sorted(picked, key=lambda p: p["id"])


def build():
    problems = load_problems(PROJECT_ROOT / "data" / "test")
    eligible_ids = set.intersection(*(answered_ids(s) for s in LIVE_SLUGS))
    eligible = [p for p in problems if p["id"] in eligible_ids]
    picked = stratified_draw(eligible, SUBSET_N, SEED)

    def dist(key):
        return dict(sorted(Counter(str(p[key]) for p in picked).items()))

    return {
        "n": len(picked),
        "seed": SEED,
        "drawn_from": "data/test",
        "n_test": len(problems),
        "eligibility": (
            "answered with a readable point estimate by every currently-served "
            "endpoint: " + ", ".join(LIVE_SLUGS)
        ),
        "n_eligible": len(eligible),
        "stratification": "proportional over (category, difficulty), floor of 1 per cell",
        "by_category": dist("category"),
        "by_difficulty": dist("difficulty"),
        "problem_ids": [p["id"] for p in picked],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the committed subset redraws identically")
    args = ap.parse_args()

    built = build()
    if args.check:
        if not OUT.exists():
            print("MISSING: data/robustness_subset.json", file=sys.stderr)
            return 1
        with open(OUT) as f:
            committed = json.load(f)
        same = committed.get("problem_ids") == built["problem_ids"]
        print(("OK: " if same else "DRIFT: ") + "committed subset "
              + ("redraws identically" if same else "does NOT match a fresh draw"))
        return 0 if same else 1

    with open(OUT, "w") as f:
        json.dump(built, f, indent=2)
    print(f"{built['n']} problems drawn from {built['n_eligible']} eligible "
          f"(of {built['n_test']} in the test split), seed {built['seed']}")
    print(f"  by category:   {built['by_category']}")
    print(f"  by difficulty: {built['by_difficulty']}")
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
