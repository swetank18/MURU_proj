"""Tests for evaluation/calibration_metrics.py.

These pin the properties the metrics are being added *for*: that the interval
score cannot be gamed by widening, that a perfectly calibrated model does not
get charged for finite-sample noise, and that the Murphy decomposition
actually reconstructs the Brier score it claims to decompose.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.calibration_metrics import (
    auroc,
    brier_decomposition,
    ece,
    ece_binning_sensitivity,
    ece_debiased,
    interval_score,
    relative_width,
    sharpness_summary,
)


# ── Interval score ─────────────────────────────────────────────────────

def test_interval_score_is_width_when_covered():
    assert abs(interval_score(0.9, 1.1, 1.0, alpha=0.1) - 0.2) < 1e-12


def test_hedging_is_penalised_like_a_miss():
    # This is the property Accuracy@CI lacks: a 100x-wide interval that covers
    # must not beat a tight one, and here it scores worse than a near miss.
    tight_hit = interval_score(0.9, 1.1, 1.0, alpha=0.1)
    wide_hedge = interval_score(-10, 10, 1.0, alpha=0.1)
    tight_miss = interval_score(2.0, 2.1, 1.0, alpha=0.1)
    assert tight_hit < tight_miss
    assert wide_hedge > tight_hit * 50
    assert abs(wide_hedge - tight_miss) < 1.0  # comparable cost


def test_miss_penalty_scales_with_confidence_level():
    # A 95% interval is charged more for missing than a 90% one.
    assert interval_score(0, 1, 2.0, alpha=0.05) > interval_score(0, 1, 2.0, alpha=0.1)


def test_inverted_interval_is_normalised():
    assert interval_score(1.1, 0.9, 1.0, alpha=0.1) == interval_score(0.9, 1.1, 1.0, alpha=0.1)


def test_relative_width():
    assert relative_width(0.0, 2.0, gt_width=0.5) == 4.0


# ── ECE ────────────────────────────────────────────────────────────────

def test_perfect_calibration_has_near_zero_debiased_ece():
    rng = random.Random(11)
    conf = [rng.choice([0.5, 0.6, 0.7, 0.8, 0.9]) for _ in range(300)]
    correct = [1 if rng.random() < c else 0 for c in conf]
    d = ece_debiased(correct, conf, n_sim=300)
    # The raw ECE is nonzero purely from finite-n scatter; debiasing removes it.
    assert d["observed"] > 0
    assert d["null_floor"] > 0
    assert d["debiased"] < 0.02


def test_overconfidence_survives_debiasing():
    conf = [0.95] * 300
    correct = [1] * 150 + [0] * 150
    d = ece_debiased(correct, conf, n_sim=300)
    assert d["debiased"] > 0.4


def test_equal_mass_binning_keeps_ties_together():
    # Verbalized confidence is heavily tied; a tied block split across bins
    # would manufacture calibration signal. Constant confidence must give a
    # single well-defined gap, not an artifact of bin edges.
    conf = [0.9] * 100
    correct = [1] * 50 + [0] * 50
    assert abs(ece(correct, conf, scheme="equal_mass") - 0.4) < 1e-9


def test_binning_sensitivity_reports_a_spread():
    rng = random.Random(3)
    conf = [rng.uniform(0.3, 1.0) for _ in range(200)]
    correct = [1 if rng.random() < c * 0.7 else 0 for c in conf]
    s = ece_binning_sensitivity(correct, conf)
    assert s["spread"] == s["max"] - s["min"] >= 0
    assert len(s["grid"]) == 8  # 2 schemes x 4 bin counts


# ── Brier / Murphy ─────────────────────────────────────────────────────

def test_murphy_decomposition_reconstructs_brier():
    rng = random.Random(5)
    conf = [rng.choice([0.2, 0.4, 0.6, 0.8, 0.95]) for _ in range(400)]
    correct = [1 if rng.random() < c else 0 for c in conf]
    d = brier_decomposition(correct, conf)
    reconstructed = d["reliability"] - d["resolution"] + d["uncertainty"]
    assert abs(reconstructed - d["brier"]) < 1e-9


def test_uninformative_model_has_zero_resolution():
    # Constant confidence carries no information about which items are right.
    conf = [0.7] * 100
    correct = [1] * 70 + [0] * 30
    assert brier_decomposition(correct, conf)["resolution"] < 1e-9


# ── AUROC ──────────────────────────────────────────────────────────────

def test_auroc_perfect_and_constant():
    correct = [1, 1, 1, 0, 0, 0]
    assert auroc(correct, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]) == 1.0
    assert auroc(correct, [0.5] * 6) == 0.5  # all ties -> chance


def test_auroc_is_undefined_without_both_outcomes():
    assert auroc([1, 1, 1], [0.9, 0.8, 0.7]) is None


# ── Sharpness summary ──────────────────────────────────────────────────

def test_sharpness_summary_reports_tail_not_just_mean():
    records = [
        {"lower": 0.9, "upper": 1.1, "gt_point": 1.0, "gt_width": 0.2, "alpha": 0.1}
    ] * 9
    records.append(
        {"lower": -500, "upper": 500, "gt_point": 1.0, "gt_width": 0.2, "alpha": 0.1}
    )
    s = sharpness_summary(records)
    assert abs(s["median_relative_width"] - 1.0) < 1e-9   # typical behaviour is sharp
    assert s["mean_relative_width"] > 100           # mean is eaten by one hedge
    assert s["hedge_rate"] == 0.1
    assert s["pe_coverage"] == 1.0


def test_missing_interval_is_counted_but_not_scored():
    records = [
        {"lower": 0.9, "upper": 1.1, "gt_point": 1.0, "gt_width": 0.2, "alpha": 0.1},
        {"lower": None, "upper": None, "gt_point": 1.0, "gt_width": 0.2, "alpha": 0.1},
    ]
    s = sharpness_summary(records)
    assert s["n_records"] == 2
    assert s["n_with_interval"] == 1
    assert s["interval_rate"] == 0.5
