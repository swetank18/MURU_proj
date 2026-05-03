"""Tests for evaluation/metrics.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import MURUMetrics, Prediction


def make_problem(pid, category="bayesian_updating", difficulty=1,
                 pe=0.5, ci=(0.4, 0.6), framework="bayesian_inference"):
    return {
        "id": pid,
        "category": category,
        "difficulty": difficulty,
        "required_framework": framework,
        "ground_truth": {
            "point_estimate": pe,
            "confidence_interval": list(ci),
            "ci_level": 0.95,
        },
    }


def test_accuracy_all_correct():
    problems = [make_problem(f"P{i}", pe=0.5, ci=(0.4, 0.6)) for i in range(10)]
    preds = [Prediction(problem_id=f"P{i}", predicted_answer=0.5,
                        predicted_confidence=0.8) for i in range(10)]
    m = MURUMetrics(problems, preds)
    assert m.accuracy() == 1.0


def test_accuracy_all_wrong():
    problems = [make_problem(f"P{i}", pe=0.5, ci=(0.4, 0.6)) for i in range(10)]
    preds = [Prediction(problem_id=f"P{i}", predicted_answer=10.0,
                        predicted_confidence=0.8) for i in range(10)]
    m = MURUMetrics(problems, preds)
    assert m.accuracy() == 0.0


def test_accuracy_at_ci_boundary():
    """Predictions at the exact CI boundary should count as correct."""
    problems = [make_problem("P1", ci=(0.4, 0.6))]
    preds_low = [Prediction(problem_id="P1", predicted_answer=0.4, predicted_confidence=0.5)]
    preds_high = [Prediction(problem_id="P1", predicted_answer=0.6, predicted_confidence=0.5)]
    assert MURUMetrics(problems, preds_low).accuracy() == 1.0
    assert MURUMetrics(problems, preds_high).accuracy() == 1.0


def test_ece_perfect_calibration():
    """If accuracy in each bin equals confidence in that bin, ECE = 0."""
    problems = [make_problem(f"P{i}", ci=(0.4, 0.6)) for i in range(10)]
    preds = []
    for i in range(10):
        # All predictions at confidence 0.5; 5 correct, 5 wrong → bin acc = 0.5
        answer = 0.5 if i < 5 else 100.0
        preds.append(Prediction(problem_id=f"P{i}", predicted_answer=answer,
                                predicted_confidence=0.5))
    m = MURUMetrics(problems, preds)
    assert m.ece() == pytest.approx(0.0, abs=0.01)


def test_ece_max_miscalibration():
    """High confidence on all wrong answers → large ECE."""
    problems = [make_problem(f"P{i}", ci=(0.4, 0.6)) for i in range(10)]
    preds = [Prediction(problem_id=f"P{i}", predicted_answer=100.0,
                        predicted_confidence=0.95) for i in range(10)]
    m = MURUMetrics(problems, preds)
    # confidence 0.95, accuracy 0 → bin contribution ≈ 0.95
    assert m.ece() > 0.9


def test_overconfidence_rate():
    """Overconfidence = high confidence (>0.7) AND wrong."""
    problems = [make_problem(f"P{i}", ci=(0.4, 0.6)) for i in range(4)]
    preds = [
        Prediction(problem_id="P0", predicted_answer=100.0, predicted_confidence=0.9),  # over
        Prediction(problem_id="P1", predicted_answer=100.0, predicted_confidence=0.5),  # wrong but low conf
        Prediction(problem_id="P2", predicted_answer=0.5, predicted_confidence=0.9),    # right
        Prediction(problem_id="P3", predicted_answer=100.0, predicted_confidence=0.71), # over
    ]
    m = MURUMetrics(problems, preds)
    assert m.overconfidence_rate() == pytest.approx(0.5)


def test_framework_match_rate():
    problems = [make_problem(f"P{i}", framework="bayesian_inference") for i in range(4)]
    preds = [
        Prediction(problem_id="P0", predicted_answer=0.5, predicted_confidence=0.5,
                   predicted_framework="bayesian_inference"),
        Prediction(problem_id="P1", predicted_answer=0.5, predicted_confidence=0.5,
                   predicted_framework="frequentist_inference"),
        Prediction(problem_id="P2", predicted_answer=0.5, predicted_confidence=0.5,
                   predicted_framework="bayesian_inference"),
        Prediction(problem_id="P3", predicted_answer=0.5, predicted_confidence=0.5,
                   predicted_framework=None),
    ]
    m = MURUMetrics(problems, preds)
    # 2 out of 4 match (None counts as no-match)
    assert m.framework_match_rate() == pytest.approx(0.5)


def test_accuracy_by_category():
    problems = [
        make_problem("P0", category="bayesian_updating", ci=(0.4, 0.6)),
        make_problem("P1", category="bayesian_updating", ci=(0.4, 0.6)),
        make_problem("P2", category="adversarial_ambiguity", ci=(0.4, 0.6)),
    ]
    preds = [
        Prediction(problem_id="P0", predicted_answer=0.5, predicted_confidence=0.5),  # correct
        Prediction(problem_id="P1", predicted_answer=10.0, predicted_confidence=0.5), # wrong
        Prediction(problem_id="P2", predicted_answer=0.5, predicted_confidence=0.5),  # correct
    ]
    m = MURUMetrics(problems, preds)
    by_cat = m.accuracy_by_category()
    assert by_cat["bayesian_updating"]["accuracy"] == pytest.approx(0.5)
    assert by_cat["adversarial_ambiguity"]["accuracy"] == pytest.approx(1.0)
    assert by_cat["bayesian_updating"]["count"] == 2


def test_accuracy_by_difficulty_monotonic_friendly():
    """Sanity: difficulty grouping returns the right counts."""
    problems = [make_problem(f"P{i}", difficulty=(i % 5) + 1, ci=(0.4, 0.6)) for i in range(10)]
    preds = [Prediction(problem_id=f"P{i}", predicted_answer=0.5, predicted_confidence=0.5)
             for i in range(10)]
    m = MURUMetrics(problems, preds)
    by_diff = m.accuracy_by_difficulty()
    for d in range(1, 6):
        assert by_diff[d]["count"] == 2
        assert by_diff[d]["accuracy"] == 1.0


def test_empty_results():
    m = MURUMetrics([], [])
    assert m.accuracy() == 0.0
    assert m.ece() == 0.0
    assert m.overconfidence_rate() == 0.0
    assert m.framework_match_rate() == 0.0


def test_prediction_for_missing_problem_is_skipped():
    problems = [make_problem("P0", ci=(0.4, 0.6))]
    preds = [
        Prediction(problem_id="P0", predicted_answer=0.5, predicted_confidence=0.5),
        Prediction(problem_id="P_nonexistent", predicted_answer=0.5, predicted_confidence=0.5),
    ]
    m = MURUMetrics(problems, preds)
    # Only the matched prediction is evaluated
    assert len(m.results) == 1
    assert m.accuracy() == 1.0


def test_compute_all_returns_all_keys():
    problems = [make_problem("P0", ci=(0.4, 0.6))]
    preds = [Prediction(problem_id="P0", predicted_answer=0.5, predicted_confidence=0.5)]
    m = MURUMetrics(problems, preds)
    result = m.compute_all()
    expected_keys = {"accuracy", "ece", "overconfidence_rate", "framework_match_rate",
                     "accuracy_by_category", "accuracy_by_difficulty", "calibration_curve",
                     "total_evaluated"}
    assert expected_keys.issubset(result.keys())
