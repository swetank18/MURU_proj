#!/usr/bin/env python3
"""
split_data.py — MURU-BENCH Data Splitter

Splits the full problem set into train/validation/test splits (80/10/10)
with stratification by category and difficulty.

Also populates the by_category/ and by_difficulty/ views with symlinks.

Usage:
    python scripts/split_data.py                     # split from a flat data/all/ directory
    python scripts/split_data.py --source data/all/  # explicit source
    python scripts/split_data.py --dry-run            # preview without moving files
    python scripts/split_data.py --seed 42            # reproducible split
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SPLIT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}

CATEGORIES = [
    "bayesian_updating",
    "conditional_probability_chains",
    "distribution_estimation",
    "decision_under_uncertainty",
    "adversarial_ambiguity",
]


def load_problems(source_dir: Path) -> list[tuple[Path, dict]]:
    """Load all problems from source directory."""
    problems = []
    for filepath in sorted(source_dir.rglob("MURU-*.json")):
        try:
            with open(filepath) as f:
                data = json.load(f)
                problems.append((filepath, data))
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Skipping {filepath}: {e}", file=sys.stderr)
    return problems


def stratified_split(
    problems: list[tuple[Path, dict]],
    seed: int = 42,
) -> dict[str, list[tuple[Path, dict]]]:
    """Split problems with stratification by category + difficulty."""
    random.seed(seed)

    # Group by (category, difficulty)
    groups = defaultdict(list)
    for filepath, data in problems:
        key = (data["category"], data["difficulty"])
        groups[key].append((filepath, data))

    splits = {"train": [], "validation": [], "test": []}

    for key, group in sorted(groups.items()):
        random.shuffle(group)
        n = len(group)
        n_val = max(1, round(n * SPLIT_RATIOS["validation"]))
        n_test = max(1, round(n * SPLIT_RATIOS["test"]))
        n_train = n - n_val - n_test

        if n_train < 0:
            # Very small groups: just put everything in train
            n_train = n
            n_val = 0
            n_test = 0

        splits["train"].extend(group[:n_train])
        splits["validation"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])

    return splits


def execute_split(
    splits: dict[str, list[tuple[Path, dict]]],
    dry_run: bool = False,
):
    """Move files into split directories and create category/difficulty views."""

    for split_name, items in splits.items():
        split_dir = DATA_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for src_path, data in items:
            dst_path = split_dir / f"{data['id']}.json"
            if dry_run:
                print(f"  [DRY RUN] {src_path.name} → {split_name}/")
            else:
                # Skip if source and dest are the same file
                if src_path.resolve() == dst_path.resolve():
                    continue
                shutil.copy2(src_path, dst_path)

    # Remove source files that were copied to different splits
    if not dry_run:
        for split_name, items in splits.items():
            split_dir = DATA_DIR / split_name
            for src_path, data in items:
                dst_path = split_dir / f"{data['id']}.json"
                if src_path.resolve() != dst_path.resolve() and src_path.exists():
                    src_path.unlink()

    # Clean up by_category and by_difficulty before recreating
    for view_dir in [DATA_DIR / "by_category", DATA_DIR / "by_difficulty"]:
        if view_dir.exists():
            shutil.rmtree(view_dir)

    # Create by_category views (symlinks to split files)
    for cat in CATEGORIES:
        cat_dir = DATA_DIR / "by_category" / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        for _, data in items:
            cat = data["category"]
            src = DATA_DIR / split_name / f"{data['id']}.json"
            dst = DATA_DIR / "by_category" / cat / f"{data['id']}.json"
            if not dry_run and src.exists() and not dst.exists():
                try:
                    os.symlink(src.resolve(), dst)
                except OSError:
                    shutil.copy2(src, dst)

    # Create by_difficulty views
    for d in range(1, 6):
        diff_dir = DATA_DIR / "by_difficulty" / str(d)
        diff_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        for _, data in items:
            diff = data["difficulty"]
            src = DATA_DIR / split_name / f"{data['id']}.json"
            dst = DATA_DIR / "by_difficulty" / str(diff) / f"{data['id']}.json"
            if not dry_run and src.exists() and not dst.exists():
                try:
                    os.symlink(src.resolve(), dst)
                except OSError:
                    shutil.copy2(src, dst)

    if not dry_run:
        print(f"\n  ✓ Split complete!")


def main():
    parser = argparse.ArgumentParser(description="Split MURU-BENCH problems into train/val/test.")
    parser.add_argument(
        "--source", type=str, default=str(DATA_DIR / "all"),
        help="Source directory with all problems (default: data/all/)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without moving files.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Source directory not found: {source}")
        print(f"Place all MURU-*.json files in {source} before splitting.")
        sys.exit(1)

    problems = load_problems(source)
    if not problems:
        print(f"No MURU-*.json files found in {source}")
        sys.exit(1)

    print(f"\n  Found {len(problems)} problems in {source}")
    splits = stratified_split(problems, seed=args.seed)

    for name, items in splits.items():
        print(f"    {name}: {len(items)} problems")

    execute_split(splits, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
