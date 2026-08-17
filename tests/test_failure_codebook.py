"""Tests for evaluation/failure_codebook.py.

The mechanical codes exist so that a share of the corpus can be coded by a rule
rather than by judgment, and the value of that is entirely in the rules being
narrow. A rule that over-fires would inflate the taxonomy's coverage and shrink
the ``_uncoded`` denominator that keeps it honest, so most of these tests are
negative cases: the near-miss that must *not* be coded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.failure_codebook import (
    MECHANICAL_CODES,
    code_mechanical,
    guaranteed_value,
    interval_covers_truth,
    matrix,
)

TWO_OPTION = (
    "A decision-maker must choose between two options. Option A (planting "
    "wheat) yields a guaranteed return of $53K. Option B (planting corn) "
    "depends on rainfall: with probability 30-54% (uncertain), it yields "
    "$90K, otherwise $12K. Compute the expected value of each option."
)


def record(**kw):
    base = {
        "model": "M",
        "problem_id": "MURU-0001",
        "stem": "",
        "gt_point": 39.3,
        "gt_ci": [30.1, 48.6],
        "model_point_estimate": None,
        "model_ci": None,
        "parse_status": "ok",
        "unit_status": "incorrect",
    }
    base.update(kw)
    return base


# --- R2: reported the other admissible answer ------------------------------


def test_r2_fires_on_the_guaranteed_option():
    r = record(stem=TWO_OPTION, model_point_estimate=53.0)
    assert "R2" in code_mechanical(r)


def test_r2_fires_when_the_same_answer_is_given_in_dollars():
    r = record(stem=TWO_OPTION, model_point_estimate=53000.0)
    assert "R2" in code_mechanical(r)


def test_r2_does_not_fire_on_a_merely_wrong_number():
    """The whole point is that the model reported *the stated payoff*."""
    r = record(stem=TWO_OPTION, model_point_estimate=47.0)
    assert "R2" not in code_mechanical(r)


def test_r2_needs_the_template():
    r = record(stem="Estimate the posterior probability.", model_point_estimate=53.0)
    assert "R2" not in code_mechanical(r)
    assert guaranteed_value("Estimate the posterior probability.") is None


def test_guaranteed_value_reads_the_stated_payoff():
    assert guaranteed_value(TWO_OPTION) == 53.0


# --- R3: reported an endpoint of its own interval ---------------------------


def test_r3_fires_at_an_endpoint():
    r = record(model_point_estimate=0.184, model_ci=[0.184, 0.342])
    assert "R3" in code_mechanical(r)


def test_r3_does_not_fire_on_a_central_estimate():
    r = record(model_point_estimate=0.263, model_ci=[0.184, 0.342])
    assert "R3" not in code_mechanical(r)


def test_r3_yields_to_m2_on_a_degenerate_interval():
    """A zero-width interval is false precision, not a summary choice."""
    codes = code_mechanical(record(model_point_estimate=5.0, model_ci=[5.0, 5.0]))
    assert "M2" in codes and "R3" not in codes


def test_r3_is_insensitive_to_interval_order():
    r = record(model_point_estimate=0.342, model_ci=[0.342, 0.184])
    assert "R3" in code_mechanical(r)


# --- M2, R4, S1 -------------------------------------------------------------


def test_m2_fires_on_zero_width():
    assert "M2" in code_mechanical(record(model_point_estimate=-12.0, model_ci=[-12.0, -12.0]))


def test_r4_mirrors_the_unit_accounting_verdict():
    assert "R4" in code_mechanical(record(unit_status="unit_mismatch"))
    assert "R4" not in code_mechanical(record(unit_status="incorrect"))


def test_s1_fires_only_on_schema_failures():
    assert "S1" in code_mechanical(record(parse_status="format_variant"))
    assert "S1" in code_mechanical(record(parse_status="no_schema"))
    assert "S1" not in code_mechanical(record(parse_status="ok"))
    # A truncated response has no answer to be off-schema about.
    assert "S1" not in code_mechanical(record(parse_status="truncated"))


def test_an_ordinary_wrong_answer_carries_no_code():
    """The uncoded share is the honest denominator; it must not be eroded."""
    r = record(model_point_estimate=0.263, model_ci=[0.184, 0.342])
    assert code_mechanical(r) == []


# --- the cross-cutting measurement -----------------------------------------


def test_interval_covers_truth():
    assert interval_covers_truth(record(model_ci=[30.0, 45.0], gt_point=39.3)) is True
    assert interval_covers_truth(record(model_ci=[10.0, 20.0], gt_point=39.3)) is False


def test_interval_covers_truth_is_undefined_without_a_usable_interval():
    assert interval_covers_truth(record(model_ci=None)) is None
    assert interval_covers_truth(record(model_ci=[5.0, 5.0], gt_point=5.0)) is None
    assert interval_covers_truth(record(model_ci=[30.0, 45.0], gt_point=None)) is None


# --- the matrix -------------------------------------------------------------


def test_matrix_counts_uncoded_against_the_model_total():
    records = [
        record(model="A", unit_status="unit_mismatch"),
        record(model="A", model_point_estimate=0.263, model_ci=[0.184, 0.342]),
        record(model="B", model_point_estimate=1.0, model_ci=[1.0, 1.0]),
    ]

    out = matrix(records)

    assert out["A"]["_errors"] == 2
    assert out["A"]["R4"] == 1
    assert out["A"]["_uncoded"] == 1
    assert out["B"]["_uncoded"] == 0
    assert all(code in out["A"] for code in MECHANICAL_CODES)


def test_matrix_counts_a_multi_coded_record_once_as_coded():
    r = record(stem=TWO_OPTION, model_point_estimate=53.0, model_ci=[53.0, 53.0])
    out = matrix([r])
    assert out["M"]["R2"] == 1 and out["M"]["M2"] == 1
    assert out["M"]["_errors"] == 1 and out["M"]["_uncoded"] == 0
