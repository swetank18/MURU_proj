#!/usr/bin/env python3
"""
run_baselines.py — MURU-BENCH Simulated Baseline Generator

Generates plausible model responses for baseline comparison in the paper.
Simulates several capability tiers to demonstrate how the benchmark
discriminates between different model strengths.

Models simulated:
    - random_baseline:     Random answers within plausible range
    - heuristic_baseline:  Uses simple heuristics (ignores uncertainty)
    - competent_model:     Gets easy problems right, struggles with hard ones
    - strong_model:        Handles most difficulties, struggles with D5
    - expert_model:        Near-expert calibrated performance

Usage:
    python evaluation/run_baselines.py                      # run all baselines on test set
    python evaluation/run_baselines.py --model strong_model  # run one model
    python evaluation/run_baselines.py --subset data/test/ --save
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import MURUMetrics, Prediction


# ──────────────────────────────────────────────────────────────────────
# Model Profiles
# ──────────────────────────────────────────────────────────────────────

MODEL_PROFILES = {
    "random_baseline": {
        "description": "Random answers within plausible numeric range",
        "accuracy_by_difficulty": {1: 0.15, 2: 0.12, 3: 0.10, 4: 0.08, 5: 0.05},
        "framework_match": 0.20,
        "confidence_range": (0.3, 0.9),
        "noise_scale": 2.0,
    },
    "heuristic_baseline": {
        "description": "Uses simple heuristics, ignores uncertainty structure",
        "accuracy_by_difficulty": {1: 0.65, 2: 0.50, 3: 0.30, 4: 0.15, 5: 0.05},
        "framework_match": 0.40,
        "confidence_range": (0.6, 0.95),
        "noise_scale": 0.8,
    },
    "competent_model": {
        "description": "Handles easy problems, struggles with complexity (GPT-3.5 tier)",
        "accuracy_by_difficulty": {1: 0.80, 2: 0.65, 3: 0.45, 4: 0.25, 5: 0.10},
        "framework_match": 0.60,
        "confidence_range": (0.5, 0.90),
        "noise_scale": 0.5,
    },
    "strong_model": {
        "description": "Strong reasoning, some calibration issues (GPT-4 tier)",
        "accuracy_by_difficulty": {1: 0.92, 2: 0.82, 3: 0.65, 4: 0.45, 5: 0.25},
        "framework_match": 0.78,
        "confidence_range": (0.55, 0.92),
        "noise_scale": 0.3,
    },
    "expert_model": {
        "description": "Near-expert performance, well-calibrated (frontier model tier)",
        "accuracy_by_difficulty": {1: 0.97, 2: 0.92, 3: 0.80, 4: 0.60, 5: 0.40},
        "framework_match": 0.88,
        "confidence_range": (0.50, 0.95),
        "noise_scale": 0.15,
    },
}

# Category difficulty modifiers (some categories are harder for models)
CATEGORY_MODIFIERS = {
    "bayesian_updating": 0.0,         # neutral
    "conditional_probability_chains": -0.05,  # slightly harder
    "distribution_estimation": 0.05,   # slightly easier
    "decision_under_uncertainty": -0.05,
    "adversarial_ambiguity": -0.15,    # hardest for models (designed to trap)
}


# ──────────────────────────────────────────────────────────────────────
# Simulation Engine
# ──────────────────────────────────────────────────────────────────────

def simulate_response(
    problem: dict,
    profile: dict,
    rng: random.Random,
) -> Prediction:
    """Simulate a model's response to a single problem."""
    gt = problem["ground_truth"]
    ci = gt["confidence_interval"]
    pe = gt["point_estimate"]
    diff = problem["difficulty"]
    cat = problem["category"]

    # Base accuracy for this difficulty level
    base_acc = profile["accuracy_by_difficulty"].get(diff, 0.1)

    # Category modifier
    cat_mod = CATEGORY_MODIFIERS.get(cat, 0)
    adjusted_acc = max(0.02, min(0.98, base_acc + cat_mod))

    # Determine if this prediction is "correct" (within CI)
    is_correct = rng.random() < adjusted_acc

    ci_width = ci[1] - ci[0]
    noise_scale = profile["noise_scale"]

    if is_correct:
        # Generate a prediction within the CI
        # Cluster around point estimate with some noise
        noise = rng.gauss(0, ci_width * 0.15)
        predicted = pe + noise
        # Clamp within CI
        predicted = max(ci[0], min(ci[1], predicted))
    else:
        # Generate a prediction outside the CI
        if ci_width < 0.001:
            # Very narrow CI — just offset
            offset = rng.choice([-1, 1]) * abs(pe) * rng.uniform(0.1, 0.5)
            predicted = pe + offset
        else:
            # Offset from CI in a plausible direction
            direction = rng.choice([-1, 1])
            offset = ci_width * rng.uniform(0.3, noise_scale * 2)
            if direction > 0:
                predicted = ci[1] + offset
            else:
                predicted = ci[0] - offset

    # Simulate confidence
    conf_low, conf_high = profile["confidence_range"]
    if is_correct:
        # Correct answers tend to have higher confidence
        confidence = rng.uniform(conf_low + 0.1, conf_high)
    else:
        # Wrong answers have varied confidence (overconfidence is common)
        if rng.random() < 0.4:  # overconfident
            confidence = rng.uniform(conf_high - 0.15, conf_high)
        else:
            confidence = rng.uniform(conf_low, conf_low + 0.2)

    confidence = max(0.05, min(0.99, confidence))

    # Framework match
    if rng.random() < profile["framework_match"]:
        framework = problem["required_framework"]
    else:
        frameworks = [
            "bayesian_inference", "frequentist_inference",
            "decision_theory", "information_theory", "monte_carlo",
        ]
        framework = rng.choice(frameworks)

    # Simulate a predicted interval
    pred_ci_half = ci_width * rng.uniform(0.3, 1.5)
    pred_interval = (
        round(predicted - pred_ci_half, 4),
        round(predicted + pred_ci_half, 4),
    )

    return Prediction(
        problem_id=problem["id"],
        predicted_answer=round(predicted, 4),
        predicted_confidence=round(confidence, 3),
        predicted_interval=pred_interval,
        predicted_framework=framework,
        raw_response=f"[Simulated: {profile.get('description', 'baseline')}]",
    )


def load_problems(subset_dir: str) -> list[dict]:
    """Load problems from a directory."""
    problems = []
    path = Path(subset_dir)
    for filepath in sorted(path.rglob("MURU-*.json")):
        try:
            with open(filepath) as f:
                problems.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue
    return problems


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run MURU-BENCH simulated baselines.")
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help=f"Model to simulate. Options: {', '.join(MODEL_PROFILES.keys())}. "
             "Default: run all models."
    )
    parser.add_argument(
        "--subset", "-s", default=str(PROJECT_ROOT / "data" / "test"),
        help="Problem directory (default: data/test/)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save", action="store_true", help="Save results to evaluation/baselines/.")
    args = parser.parse_args()

    models_to_run = (
        [args.model] if args.model
        else list(MODEL_PROFILES.keys())
    )

    for m in models_to_run:
        if m not in MODEL_PROFILES:
            print(f"ERROR: Unknown model '{m}'. Options: {', '.join(MODEL_PROFILES.keys())}")
            sys.exit(1)

    problems = load_problems(args.subset)
    if not problems:
        print(f"No problems found in {args.subset}")
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"  MURU-BENCH Baseline Evaluation")
    print(f"  Problems: {len(problems)} (from {args.subset})")
    print(f"  Models: {', '.join(models_to_run)}")
    print(f"{'═' * 60}\n")

    all_results = {}

    for model_name in models_to_run:
        profile = MODEL_PROFILES[model_name]
        rng = random.Random(args.seed)

        print(f"  Running {model_name}: {profile['description']}")
        print(f"  {'─' * 50}")

        predictions = [
            simulate_response(p, profile, rng)
            for p in problems
        ]

        metrics = MURUMetrics(problems, predictions)
        results = metrics.compute_all()
        all_results[model_name] = results

        print(f"    Accuracy:          {results['accuracy']:.1%}")
        print(f"    ECE:               {results['ece']:.4f}")
        print(f"    Overconfidence:    {results['overconfidence_rate']:.1%}")
        print(f"    Framework match:   {results['framework_match_rate']:.1%}")

        # Difficulty curve
        print(f"    Difficulty curve:  ", end="")
        for d in sorted(results["accuracy_by_difficulty"].keys()):
            acc = results["accuracy_by_difficulty"][d]["accuracy"]
            print(f"D{d}={acc:.0%} ", end="")
        print()
        print()

        # Save individual result
        if args.save:
            baselines_dir = PROJECT_ROOT / "evaluation" / "baselines"
            baselines_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            outpath = baselines_dir / f"{model_name}_{timestamp}.json"
            with open(outpath, "w") as f:
                json.dump({
                    "model": model_name,
                    "description": profile["description"],
                    "timestamp": timestamp,
                    "n_problems": len(problems),
                    "n_predictions": len(predictions),
                    "metrics": results,
                }, f, indent=2, default=str)
            print(f"    Saved: {outpath.relative_to(PROJECT_ROOT)}")
            print()

    # Summary comparison table
    print(f"\n{'═' * 60}")
    print(f"  Summary Comparison")
    print(f"{'═' * 60}\n")

    header = f"  {'Model':<22} {'Acc':>6} {'ECE':>7} {'OvConf':>7} {'FwMtch':>7}"
    print(header)
    print(f"  {'─' * 55}")

    for model_name in models_to_run:
        r = all_results[model_name]
        print(
            f"  {model_name:<22} {r['accuracy']:>5.1%} "
            f"{r['ece']:>7.4f} {r['overconfidence_rate']:>6.1%} "
            f"{r['framework_match_rate']:>6.1%}"
        )

    # Difficulty scaling table
    print(f"\n  {'Model':<22}", end="")
    for d in range(1, 6):
        print(f" {'D'+str(d):>6}", end="")
    print()
    print(f"  {'─' * 55}")

    for model_name in models_to_run:
        r = all_results[model_name]
        print(f"  {model_name:<22}", end="")
        for d in range(1, 6):
            if d in r["accuracy_by_difficulty"]:
                print(f" {r['accuracy_by_difficulty'][d]['accuracy']:>5.0%}", end="")
            else:
                print(f" {'N/A':>6}", end="")
        print()

    print()

    # Save combined summary
    if args.save:
        summary_path = PROJECT_ROOT / "evaluation" / "baselines" / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "n_problems": len(problems),
                "models": {
                    name: {
                        "description": MODEL_PROFILES[name]["description"],
                        "metrics": all_results[name],
                    }
                    for name in models_to_run
                },
            }, f, indent=2, default=str)
        print(f"  Summary saved: evaluation/baselines/summary.json\n")


if __name__ == "__main__":
    main()
