"""Smoke tests for the dataset itself: schema conformance & invariants."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIRS = [
    PROJECT_ROOT / "data" / "train",
    PROJECT_ROOT / "data" / "validation",
    PROJECT_ROOT / "data" / "test",
]

REQUIRED_FIELDS = {
    "id", "category", "difficulty", "stem", "uncertainty_type",
    "required_framework", "ground_truth", "solution_steps",
    "common_failure_modes", "metadata",
}

VALID_CATEGORIES = {
    "bayesian_updating", "conditional_probability_chains",
    "distribution_estimation", "decision_under_uncertainty",
    "adversarial_ambiguity",
}

VALID_FRAMEWORKS = {
    "bayesian_inference", "frequentist_inference", "decision_theory",
    "information_theory", "monte_carlo",
}


def all_problems():
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.rglob("MURU-*.json")):
            with open(path) as f:
                yield path, json.load(f)


def test_problem_count_matches_3000():
    total = sum(1 for _ in all_problems())
    assert total == 3000, f"Expected 3000 problems total, got {total}"


def test_no_duplicate_ids():
    seen_ids = []
    for _, problem in all_problems():
        seen_ids.append(problem["id"])
    assert len(seen_ids) == len(set(seen_ids)), "Duplicate problem IDs detected"


def test_all_required_fields_present():
    for path, problem in all_problems():
        missing = REQUIRED_FIELDS - set(problem.keys())
        assert not missing, f"{path.name}: missing fields {missing}"


def test_categories_are_valid():
    for path, problem in all_problems():
        assert problem["category"] in VALID_CATEGORIES, (
            f"{path.name}: invalid category '{problem['category']}'"
        )


def test_difficulty_in_range():
    for path, problem in all_problems():
        assert 1 <= problem["difficulty"] <= 5, (
            f"{path.name}: difficulty {problem['difficulty']} out of [1,5]"
        )


def test_required_framework_valid():
    for path, problem in all_problems():
        assert problem["required_framework"] in VALID_FRAMEWORKS, (
            f"{path.name}: invalid framework '{problem['required_framework']}'"
        )


def test_ground_truth_ci_well_formed():
    """CI must be [lower, upper] with lower <= point_estimate <= upper."""
    for path, problem in all_problems():
        gt = problem["ground_truth"]
        ci = gt["confidence_interval"]
        pe = gt["point_estimate"]
        assert len(ci) == 2, f"{path.name}: CI must have 2 values"
        assert ci[0] <= ci[1], f"{path.name}: CI lower {ci[0]} > upper {ci[1]}"
        assert ci[0] <= pe <= ci[1], (
            f"{path.name}: point_estimate {pe} not in CI [{ci[0]}, {ci[1]}]"
        )


def test_id_format():
    for path, problem in all_problems():
        pid = problem["id"]
        assert pid.startswith("MURU-"), f"{path.name}: ID '{pid}' wrong prefix"
        assert pid[5:].isdigit(), f"{path.name}: ID '{pid}' suffix not numeric"
