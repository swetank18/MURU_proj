"""Tests for evaluation/unit_accounting.py.

The property that matters is asymmetric: the rule must credit a model that
answered correctly in dollars, and must *not* credit a model whose answer is
simply a thousand times too large. Those two look identical if you only check
whether the rescaled point estimate lands inside the ground-truth interval,
which is why the corroboration requirement exists and why most of these tests
are about the second case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.unit_accounting import (
    admissible_factors,
    classify_prediction,
    is_correct,
    report,
)


def problem(stem, point, ci, answer=""):
    return {
        "stem": stem,
        "ground_truth": {
            "point_estimate": point,
            "confidence_interval": ci,
            "answer": answer,
        },
    }


DOLLARS = problem(
    "Option A yields a guaranteed return of $178K. Option B yields $323K "
    "with probability 34-49%, otherwise $50K. Compute the expected value.",
    164.7,
    [142.8, 183.8],
)

PROBABILITY = problem(
    "The base rate is estimated between 2.00% and 4.20%. What is the "
    "probability the flagged applicant is a high performer?",
    0.254,
    [0.178, 0.318],
)


# ── Which readings the wording leaves open ─────────────────────────────

def test_thousands_marker_is_read_from_the_stem():
    assert 1000.0 in admissible_factors(DOLLARS)


def test_percent_marker_is_read_from_the_stem():
    assert 100.0 in admissible_factors(PROBABILITY)


def test_unmarked_stem_admits_no_rescale():
    p = problem("A fair coin is flipped ten times. How many heads?", 5.0, [3.0, 7.0])
    assert admissible_factors(p) == []
    # ...so a factor-of-1000 answer stays wrong, corroborating interval or not.
    assert classify_prediction(5000.0, [3000.0, 7000.0], p) == "incorrect"


# ── The case the rule exists to credit ─────────────────────────────────

def test_answer_in_dollars_is_a_unit_mismatch_not_an_error():
    # Real record: Llama-3.3-70B on MURU-0365. Point and interval are the
    # ground truth times a thousand.
    assert classify_prediction(163195.0, [142820.0, 183570.0], DOLLARS) == "unit_mismatch"


def test_answer_in_percent_is_a_unit_mismatch():
    assert classify_prediction(23.0, [18.0, 28.0], PROBABILITY) == "unit_mismatch"


def test_correct_answer_is_untouched():
    assert classify_prediction(164.0, [150.0, 180.0], DOLLARS) == "correct"


# ── The cases the rule must refuse ─────────────────────────────────────

def test_wrong_by_a_factor_of_a_thousand_is_not_credited():
    # Real record: Llama-3.1-8B on MURU-3022 answered 846.84 (its own units
    # were $K) against a ground truth of 10.9 [-4.6, 48.7]. Dividing by 1000
    # lands inside that wide interval purely by accident, and the model's own
    # interval does not corroborate the reading.
    gt = problem("The investment cost is $350K; it yields $1683K.", 10.9, [-4.6, 48.7])
    assert classify_prediction(846.84, [484.75, 846.84], gt) == "unit_mismatch_uncorroborated"
    assert not is_correct("unit_mismatch_uncorroborated", unit_aware=True)


def test_no_interval_means_no_corroboration():
    assert classify_prediction(163195.0, None, DOLLARS) == "unit_mismatch_uncorroborated"


def test_degenerate_interval_cannot_corroborate():
    # A model that emits [x, x] has stated a point, not a range.
    assert classify_prediction(163195.0, [163195.0, 163195.0], DOLLARS) == "unit_mismatch_uncorroborated"


def test_conflicting_interval_is_not_credited():
    # Point rescales into range; the model's own interval is somewhere else
    # entirely, so the rescale is not the model's reading.
    assert classify_prediction(163195.0, [900000.0, 950000.0], DOLLARS) == "unit_mismatch_uncorroborated"


def test_plain_wrong_stays_wrong():
    assert classify_prediction(500.0, [400.0, 600.0], DOLLARS) == "incorrect"


def test_missing_point_estimate_is_incorrect():
    assert classify_prediction(None, None, DOLLARS) == "incorrect"


# ── Accounting ─────────────────────────────────────────────────────────

def test_report_separates_the_two_accountings():
    statuses = ["correct"] * 6 + ["unit_mismatch"] * 2 + [
        "unit_mismatch_uncorroborated"
    ] + ["incorrect"]
    r = report(statuses)
    assert r["accuracy_raw"] == 0.6
    assert r["accuracy_unit_aware"] == 0.8
    assert r["accuracy_upper_bound"] == 0.9
    assert abs(r["unit_mismatch_share_of_errors"] - 0.5) < 1e-12


def test_report_on_empty_input():
    assert report([]) == {"n": 0}
