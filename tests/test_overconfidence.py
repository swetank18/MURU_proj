"""Tests for evaluation/overconfidence.py.

These pin the properties the threshold sensitivity was added to expose: that
the tie convention is what moves the number when a cut lands on a spike, that
the rate factors into an accuracy term and a metacognitive term, and that the
panel-level verdict refuses to certify a cut where the statistic has stopped
counting anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.overconfidence import (
    CANONICAL_THRESHOLD,
    confident_given_error,
    confidence_on_error,
    high_confidence_error_rate,
    mean_overconfidence,
    overconfidence_rate,
    panel_cut_dependence,
    signed_calibration_gap,
    spearman,
    summary,
    threshold_sweep,
    tie_mass,
)


# ── The cut and its tie convention ─────────────────────────────────────

def test_canonical_cut_is_strict():
    # Two wrong answers stated at exactly 0.7. Strict excludes them, inclusive
    # counts them: the published convention is strict, and the difference has
    # to be visible rather than a rounding accident.
    correct = [0, 0]
    conf = [0.7, 0.7]
    assert overconfidence_rate(correct, conf, 0.7, inclusive=False) == 0.0
    assert overconfidence_rate(correct, conf, 0.7, inclusive=True) == 1.0


def test_correct_answers_are_never_overconfident():
    assert overconfidence_rate([1, 1, 1], [1.0, 1.0, 1.0]) == 0.0


def test_tie_mass_finds_the_spike():
    conf = [0.95] * 7 + [0.5, 0.8, 0.9]
    assert tie_mass(conf, 0.95) == 0.7
    assert tie_mass(conf, 0.7) == 0.0


def test_sweep_is_monotone_in_the_threshold():
    # Raising the bar can only remove confident errors, never add them.
    correct = [0, 0, 1, 0, 1, 0]
    conf = [0.55, 0.75, 0.9, 0.95, 0.6, 1.0]
    rates = [r["rate_strict"] for r in threshold_sweep(correct, conf)]
    assert rates == sorted(rates, reverse=True)


# ── Population vs conditional reading ──────────────────────────────────

def test_population_and_conditional_rates_differ():
    # Two confident errors out of ten answers, but only three confident
    # answers in total: 20% of the split, 67% of what a user would act on.
    correct = [0, 0, 1] + [1] * 7
    conf = [0.9, 0.9, 0.9] + [0.3] * 7
    assert abs(overconfidence_rate(correct, conf) - 0.2) < 1e-12
    rate, n = high_confidence_error_rate(correct, conf)
    assert n == 3
    assert abs(rate - 2 / 3) < 1e-12


def test_conditional_rate_is_undefined_without_confident_answers():
    assert high_confidence_error_rate([0, 1], [0.1, 0.2]) == (None, 0)


# ── The factorisation that separates accuracy from metacognition ───────

def test_rate_factors_into_error_rate_times_confident_given_error():
    correct = [0, 1, 0, 1, 0, 1, 1, 0]
    conf = [0.9, 0.9, 0.5, 0.95, 0.8, 0.2, 0.9, 0.75]
    s = summary(correct, conf)
    assert abs(s["rate"] - s["error_rate"] * s["confident_given_error"]) < 1e-12


def test_confident_given_error_ignores_accuracy():
    # Same metacognition (every error asserted confidently), very different
    # accuracy: the rate moves, the metacognitive term does not.
    weak = ([0] * 8 + [1] * 2, [0.9] * 10)
    strong = ([0] * 2 + [1] * 8, [0.9] * 10)
    assert overconfidence_rate(*weak) == 0.8
    assert overconfidence_rate(*strong) == 0.2
    assert confident_given_error(*weak)[0] == confident_given_error(*strong)[0] == 1.0


# ── Threshold-free companions ──────────────────────────────────────────

def test_mean_overconfidence_needs_no_cut():
    # Confidence carried by the wrong answers, averaged over the split: two
    # errors at 0.8 over four answers.
    assert abs(mean_overconfidence([0, 0, 1, 1], [0.8, 0.8, 1.0, 1.0]) - 0.4) < 1e-12


def test_mean_overconfidence_is_blind_to_the_threshold_but_not_the_magnitude():
    # Both models state every error confidently *past any cut below 0.6*, so
    # the rate ties them; the one asserting them harder scores worse here.
    mild = mean_overconfidence([0, 0, 1, 1], [0.6, 0.6, 0.5, 0.5])
    hard = mean_overconfidence([0, 0, 1, 1], [1.0, 1.0, 0.5, 0.5])
    assert hard > mild


def test_signed_gap_reports_direction():
    assert signed_calibration_gap([0, 0, 1, 1], [0.9, 0.9, 0.9, 0.9]) > 0  # over
    assert signed_calibration_gap([1, 1, 1, 1], [0.6, 0.6, 0.6, 0.6]) < 0  # under


def test_confidence_on_error_is_none_when_nothing_is_wrong():
    assert confidence_on_error([1, 1], [0.9, 0.9]) is None


# ── Rank correlation ───────────────────────────────────────────────────

def test_spearman_handles_ties_and_reversals():
    assert spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert spearman([1, 1, 1], [1, 2, 3]) is None  # no variance to rank


# ── Panel-level cut dependence ─────────────────────────────────────────

def _sweep(correct, conf):
    return threshold_sweep(correct, conf)


def test_panel_ordering_is_stable_when_models_only_shift_in_level():
    # Three models whose confident-error rates differ by level but whose
    # confidences all sit in the same place: no cut should reorder them.
    n = 40
    panel = {
        "weak": _sweep([0] * 30 + [1] * 10, [0.9] * n),
        "mid": _sweep([0] * 20 + [1] * 20, [0.9] * n),
        "strong": _sweep([0] * 12 + [1] * 28, [0.9] * n),
    }
    out = panel_cut_dependence(panel)
    assert out["orderings_agree"] is True
    assert out["min_rank_corr_informative"] == 1.0


def test_cut_on_a_spike_is_flagged_degenerate():
    # Every model states 0.95 on almost everything. A strict cut at 0.95
    # throws that mass away, so the cut is not a measurement of the models.
    n = 40
    panel = {
        "a": _sweep([0] * 20 + [1] * 20, [0.95] * n),
        "b": _sweep([0] * 10 + [1] * 30, [0.95] * n),
    }
    out = panel_cut_dependence(panel)
    assert 0.95 in out["degenerate_cuts"]
    assert 0.95 not in out["informative_range"]


def test_thin_cut_is_flagged_degenerate():
    # At 0.99 only a couple of confident errors survive for either model.
    panel = {
        "a": _sweep([0] * 2 + [1] * 98, [1.0] * 2 + [0.5] * 98),
        "b": _sweep([0] * 3 + [1] * 97, [1.0] * 3 + [0.5] * 97),
    }
    out = panel_cut_dependence(panel)
    assert 0.99 in out["degenerate_cuts"]


def test_ratio_is_none_rather_than_infinite_when_a_model_is_clean():
    panel = {
        "clean": _sweep([1] * 20, [0.9] * 20),
        "bad": _sweep([0] * 20, [0.9] * 20),
    }
    out = panel_cut_dependence(panel)
    at_canonical = [
        r for r in out["per_threshold"] if r["tau"] == CANONICAL_THRESHOLD
    ][0]
    assert at_canonical["min"] == 0.0
    assert at_canonical["ratio"] is None


def test_empty_panel_does_not_crash():
    out = panel_cut_dependence({})
    assert out["per_threshold"] == []
