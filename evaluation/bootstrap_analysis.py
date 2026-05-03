#!/usr/bin/env python3
"""
bootstrap_analysis.py — Statistical rigor for MURU-BENCH harness validation.

Re-runs the simulator (with the canonical seed=42) to capture per-problem
outcomes for each tier, then computes:

  (1) Bootstrap 95% CIs on Accuracy@CI, ECE, overconfidence rate,
      framework-match rate (B = 10000 resamples).
  (2) McNemar paired-comparison tests between adjacent tiers
      (Random < Heuristic < Competent < Strong < Expert) on the same 301
      problems, since all tiers see identical inputs.

Output: evaluation/baselines/bootstrap_summary.json
"""

import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import MURUMetrics, Prediction
from evaluation.run_baselines import (
    MODEL_PROFILES,
    load_problems,
    simulate_response,
)


CANONICAL_SEED = 42
N_BOOTSTRAP = 10_000
TIER_ORDER = [
    "random_baseline",
    "heuristic_baseline",
    "competent_model",
    "strong_model",
    "expert_model",
]


def per_problem_outcomes(problems, predictions):
    """Return aligned per-problem 0/1 vectors for accuracy, framework, overconf."""
    metrics = MURUMetrics(problems, predictions)
    by_id = {r.problem_id: r for r in metrics.results}
    correct, fwmatch, overconf, conf = [], [], [], []
    for p in problems:
        r = by_id[p["id"]]
        correct.append(int(r.correct))
        fwmatch.append(int(r.framework_match))
        overconf.append(int(r.overconfident))
        conf.append(r.model_confidence)
    return correct, fwmatch, overconf, conf


def bootstrap_ci(values, stat_fn, n_boot=N_BOOTSTRAP, alpha=0.05, rng=None):
    """Percentile bootstrap 95% CI for a statistic computed on `values`."""
    rng = rng or random.Random(0xC0FFEE)
    n = len(values)
    estimates = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(stat_fn(sample))
    estimates.sort()
    lo = estimates[int((alpha / 2) * n_boot)]
    hi = estimates[int((1 - alpha / 2) * n_boot)]
    point = stat_fn(values)
    return point, lo, hi


def bootstrap_paired_indices(n, n_boot, rng):
    """Yield n_boot bootstrap index sets of size n (paired across tiers)."""
    for _ in range(n_boot):
        yield [rng.randrange(n) for _ in range(n)]


def ece_from_paired(correct, conf, n_bins=10):
    """ECE on already-paired (correct, confidence) lists."""
    if not correct:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, p in zip(correct, conf):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((c, p))
    n = len(correct)
    e = 0.0
    for b in bins:
        if not b:
            continue
        bc = sum(x[0] for x in b) / len(b)
        bp = sum(x[1] for x in b) / len(b)
        e += (len(b) / n) * abs(bc - bp)
    return e


def mcnemar_test(a, b):
    """
    Exact McNemar test (binomial) on paired binary outcomes a, b.
    Returns (b_only, a_only, two_sided_p).
      - b_only: count where b is correct and a is wrong (b improves over a)
      - a_only: count where a is correct and b is wrong
    """
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # b only
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # a only
    n_disc = n01 + n10
    if n_disc == 0:
        return n01, n10, 1.0
    # Exact two-sided binomial test, p = 0.5
    k = min(n01, n10)
    # P(X <= k) for X ~ Binomial(n_disc, 0.5)
    cum = 0.0
    for i in range(k + 1):
        cum += math.comb(n_disc, i) * (0.5 ** n_disc)
    p_two = min(1.0, 2 * cum)
    return n01, n10, p_two


def main():
    test_dir = PROJECT_ROOT / "data" / "test"
    problems = load_problems(str(test_dir))
    print(f"Loaded {len(problems)} problems from {test_dir.relative_to(PROJECT_ROOT)}")

    # Step 1: regenerate predictions for each tier with the canonical seed.
    tier_outcomes = {}
    for tier in TIER_ORDER:
        rng = random.Random(CANONICAL_SEED)
        preds = [simulate_response(p, MODEL_PROFILES[tier], rng) for p in problems]
        c, f, o, conf = per_problem_outcomes(problems, preds)
        tier_outcomes[tier] = dict(correct=c, fwmatch=f, overconf=o, conf=conf)
        print(f"  {tier}: acc={sum(c)/len(c):.3f}  "
              f"fw={sum(f)/len(f):.3f}  oc={sum(o)/len(o):.3f}")

    # Step 2: bootstrap CIs (paired across tiers via shared resample indices).
    print(f"\nBootstrapping ({N_BOOTSTRAP} resamples)...")
    rng = random.Random(0xC0FFEE)
    n = len(problems)
    indices = list(bootstrap_paired_indices(n, N_BOOTSTRAP, rng))

    boot = {tier: {"acc": [], "ece": [], "ovconf": [], "fw": []} for tier in TIER_ORDER}
    for idxs in indices:
        for tier in TIER_ORDER:
            o = tier_outcomes[tier]
            c_s = [o["correct"][i] for i in idxs]
            f_s = [o["fwmatch"][i] for i in idxs]
            oc_s = [o["overconf"][i] for i in idxs]
            conf_s = [o["conf"][i] for i in idxs]
            boot[tier]["acc"].append(sum(c_s) / n)
            boot[tier]["fw"].append(sum(f_s) / n)
            boot[tier]["ovconf"].append(sum(oc_s) / n)
            boot[tier]["ece"].append(ece_from_paired(c_s, conf_s))

    summary = {"n": n, "n_bootstrap": N_BOOTSTRAP, "tiers": {}}
    for tier in TIER_ORDER:
        o = tier_outcomes[tier]
        point_acc = sum(o["correct"]) / n
        point_fw = sum(o["fwmatch"]) / n
        point_oc = sum(o["overconf"]) / n
        point_ece = ece_from_paired(o["correct"], o["conf"])

        def ci(arr):
            arr = sorted(arr)
            return arr[int(0.025 * N_BOOTSTRAP)], arr[int(0.975 * N_BOOTSTRAP)]

        summary["tiers"][tier] = {
            "accuracy": {"point": point_acc, "ci95": ci(boot[tier]["acc"])},
            "ece": {"point": point_ece, "ci95": ci(boot[tier]["ece"])},
            "overconfidence": {"point": point_oc, "ci95": ci(boot[tier]["ovconf"])},
            "framework_match": {"point": point_fw, "ci95": ci(boot[tier]["fw"])},
        }

    # Step 3: McNemar paired comparisons between adjacent tiers (accuracy@CI).
    print("\nMcNemar paired comparisons (adjacent tiers, Accuracy@CI):")
    pairs = list(zip(TIER_ORDER[:-1], TIER_ORDER[1:]))
    summary["mcnemar"] = []
    for a, b in pairs:
        ca = tier_outcomes[a]["correct"]
        cb = tier_outcomes[b]["correct"]
        n01, n10, p = mcnemar_test(ca, cb)
        diff = sum(cb) / n - sum(ca) / n
        summary["mcnemar"].append({
            "tier_a": a, "tier_b": b,
            "n_b_only": n01, "n_a_only": n10,
            "delta_accuracy": diff, "p_value": p,
        })
        print(f"  {a:>20} vs {b:<20}  Δacc={diff:+.3f}  "
              f"b-only={n01:3d}  a-only={n10:3d}  p={p:.2e}")

    # Bonus: Random vs Expert (largest gap)
    ca = tier_outcomes["random_baseline"]["correct"]
    cb = tier_outcomes["expert_model"]["correct"]
    n01, n10, p = mcnemar_test(ca, cb)
    diff = sum(cb) / n - sum(ca) / n
    summary["mcnemar"].append({
        "tier_a": "random_baseline", "tier_b": "expert_model",
        "n_b_only": n01, "n_a_only": n10,
        "delta_accuracy": diff, "p_value": p,
    })

    # Save
    out = PROJECT_ROOT / "evaluation" / "baselines" / "bootstrap_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
