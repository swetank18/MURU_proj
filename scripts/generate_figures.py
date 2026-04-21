#!/usr/bin/env python3
"""
generate_figures.py — MURU-BENCH Paper Figure Generator

Generates publication-quality figures for the NeurIPS paper appendix.

Figures generated:
    1. category_distribution.png   — Bar chart of problem counts by category
    2. difficulty_distribution.png — Bar chart of problem counts by difficulty
    3. category_difficulty_heatmap.png — Heatmap of category × difficulty matrix
    4. difficulty_scaling_curve.png — Line chart of accuracy vs difficulty by model
    5. calibration_comparison.png  — Grouped bar chart of calibration metrics

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --output paper/figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ──────────────────────────────────────────────────────────────────────
# Style configuration — NeurIPS-ready
# ──────────────────────────────────────────────────────────────────────

COLORS = {
    "bayesian_updating": "#4C72B0",
    "conditional_probability_chains": "#DD8452",
    "distribution_estimation": "#55A868",
    "decision_under_uncertainty": "#C44E52",
    "adversarial_ambiguity": "#8172B3",
}

CAT_LABELS = {
    "bayesian_updating": "Bayesian\nUpdating",
    "conditional_probability_chains": "Cond. Prob.\nChains",
    "distribution_estimation": "Distribution\nEstimation",
    "decision_under_uncertainty": "Decision Under\nUncertainty",
    "adversarial_ambiguity": "Adversarial\nAmbiguity",
}

CAT_SHORT = {
    "bayesian_updating": "BU",
    "conditional_probability_chains": "CPC",
    "distribution_estimation": "DE",
    "decision_under_uncertainty": "DUU",
    "adversarial_ambiguity": "AA",
}

MODEL_COLORS = {
    "random_baseline": "#bdbdbd",
    "heuristic_baseline": "#fdae6b",
    "competent_model": "#74c476",
    "strong_model": "#6baed6",
    "expert_model": "#9e9ac8",
}

MODEL_LABELS = {
    "random_baseline": "Random",
    "heuristic_baseline": "Heuristic",
    "competent_model": "Competent",
    "strong_model": "Strong",
    "expert_model": "Expert",
}


def setup_style():
    """Set matplotlib style for publication figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_all_problems() -> list[dict]:
    """Load all problems from all splits."""
    problems = []
    for split in ["train", "validation", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue
        for f in sorted(split_dir.glob("MURU-*.json")):
            try:
                with open(f) as fh:
                    problems.append(json.load(fh))
            except (json.JSONDecodeError, IOError):
                continue
    return problems


def load_baselines() -> dict | None:
    """Load baseline results if available."""
    summary_path = PROJECT_ROOT / "evaluation" / "baselines" / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────────────────────────────
# Figure 1: Category Distribution
# ──────────────────────────────────────────────────────────────────────

def fig_category_distribution(problems: list[dict], output_dir: Path):
    """Bar chart of problem counts by category."""
    cats = defaultdict(int)
    for p in problems:
        cats[p["category"]] += 1

    # Sort by count descending
    sorted_cats = sorted(cats.items(), key=lambda x: -x[1])
    names = [CAT_LABELS[c] for c, _ in sorted_cats]
    counts = [v for _, v in sorted_cats]
    colors = [COLORS[c] for c, _ in sorted_cats]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5, width=0.65)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            f"{count}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_ylabel("Number of Problems")
    ax.set_title("MURU-BENCH: Problem Distribution by Category", fontweight="bold", pad=15)
    ax.set_ylim(0, max(counts) * 1.12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    outpath = output_dir / "category_distribution.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath.name}")


# ──────────────────────────────────────────────────────────────────────
# Figure 2: Difficulty Distribution
# ──────────────────────────────────────────────────────────────────────

def fig_difficulty_distribution(problems: list[dict], output_dir: Path):
    """Bar chart of problem counts by difficulty level."""
    diffs = defaultdict(int)
    for p in problems:
        diffs[p["difficulty"]] += 1

    levels = sorted(diffs.keys())
    counts = [diffs[d] for d in levels]
    labels = [f"D{d}" for d in levels]

    # Gradient from green (easy) to red (hard)
    cmap = plt.cm.RdYlGn_r
    colors = [cmap(0.15 + 0.7 * (i / (len(levels) - 1))) for i in range(len(levels))]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5, width=0.55)

    for bar, count in zip(bars, counts):
        pct = count / len(problems) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{count}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Number of Problems")
    ax.set_title("MURU-BENCH: Problem Distribution by Difficulty", fontweight="bold", pad=15)
    ax.set_ylim(0, max(counts) * 1.18)

    plt.tight_layout()
    outpath = output_dir / "difficulty_distribution.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath.name}")


# ──────────────────────────────────────────────────────────────────────
# Figure 3: Category × Difficulty Heatmap
# ──────────────────────────────────────────────────────────────────────

def fig_heatmap(problems: list[dict], output_dir: Path):
    """Heatmap of category × difficulty distribution."""
    matrix = defaultdict(lambda: defaultdict(int))
    for p in problems:
        matrix[p["category"]][p["difficulty"]] += 1

    cat_order = [
        "bayesian_updating",
        "distribution_estimation",
        "decision_under_uncertainty",
        "adversarial_ambiguity",
        "conditional_probability_chains",
    ]
    diff_order = [1, 2, 3, 4, 5]

    data = np.zeros((len(cat_order), len(diff_order)))
    for i, cat in enumerate(cat_order):
        for j, diff in enumerate(diff_order):
            data[i, j] = matrix[cat][diff]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    # Labels
    ax.set_xticks(range(len(diff_order)))
    ax.set_xticklabels([f"D{d}" for d in diff_order])
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels([CAT_LABELS[c].replace("\n", " ") for c in cat_order])

    # Annotations
    for i in range(len(cat_order)):
        for j in range(len(diff_order)):
            val = int(data[i, j])
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xlabel("Difficulty Level")
    ax.set_title("Category × Difficulty Distribution", fontweight="bold", pad=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Problem Count")

    plt.tight_layout()
    outpath = output_dir / "category_difficulty_heatmap.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath.name}")


# ──────────────────────────────────────────────────────────────────────
# Figure 4: Difficulty Scaling Curves
# ──────────────────────────────────────────────────────────────────────

def fig_difficulty_curves(baselines: dict, output_dir: Path):
    """Line chart: accuracy vs difficulty by model tier."""
    fig, ax = plt.subplots(figsize=(8, 5))

    model_order = [
        "random_baseline", "heuristic_baseline",
        "competent_model", "strong_model", "expert_model",
    ]

    for model_name in model_order:
        if model_name not in baselines["models"]:
            continue
        data = baselines["models"][model_name]["metrics"]["accuracy_by_difficulty"]
        diffs = sorted(data.keys(), key=lambda x: int(x))
        accs = [data[d]["accuracy"] * 100 for d in diffs]
        diff_labels = [int(d) for d in diffs]

        ax.plot(
            diff_labels, accs,
            marker="o", markersize=8, linewidth=2.5,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Difficulty by Model Tier", fontweight="bold", pad=15)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["D1", "D2", "D3", "D4", "D5"])
    ax.set_ylim(-2, 102)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    outpath = output_dir / "difficulty_scaling_curve.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath.name}")


# ──────────────────────────────────────────────────────────────────────
# Figure 5: Calibration Comparison
# ──────────────────────────────────────────────────────────────────────

def fig_calibration(baselines: dict, output_dir: Path):
    """Grouped bar chart: ECE and overconfidence by model."""
    model_order = [
        "random_baseline", "heuristic_baseline",
        "competent_model", "strong_model", "expert_model",
    ]

    names = [MODEL_LABELS[m] for m in model_order if m in baselines["models"]]
    eces = [baselines["models"][m]["metrics"]["ece"] for m in model_order if m in baselines["models"]]
    ovconfs = [baselines["models"][m]["metrics"]["overconfidence_rate"] for m in model_order if m in baselines["models"]]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width / 2, eces, width, label="ECE ↓", color="#6baed6", edgecolor="white")
    bars2 = ax.bar(x + width / 2, ovconfs, width, label="Overconfidence ↓", color="#fc8d59", edgecolor="white")

    ax.set_ylabel("Rate")
    ax.set_title("Calibration Metrics by Model Tier", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    outpath = output_dir / "calibration_comparison.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  ✓ {outpath.name}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate MURU-BENCH paper figures.")
    parser.add_argument(
        "--output", "-o", type=str,
        default=str(PROJECT_ROOT / "paper" / "figures"),
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    setup_style()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = load_all_problems()
    baselines = load_baselines()

    print(f"\n{'═' * 60}")
    print(f"  MURU-BENCH Figure Generator")
    print(f"  Problems: {len(problems)}")
    print(f"  Output: {output_dir}")
    print(f"{'═' * 60}\n")

    # Dataset figures
    fig_category_distribution(problems, output_dir)
    fig_difficulty_distribution(problems, output_dir)
    fig_heatmap(problems, output_dir)

    # Baseline figures (only if baselines exist)
    if baselines:
        fig_difficulty_curves(baselines, output_dir)
        fig_calibration(baselines, output_dir)
    else:
        print("  ⚠ No baselines found — skipping model figures")
        print("  Run 'python evaluation/run_baselines.py --save' first")

    print(f"\n  ✓ All figures saved to {output_dir}/\n")


if __name__ == "__main__":
    main()
