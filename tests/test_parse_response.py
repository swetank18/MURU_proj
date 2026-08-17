"""Tests for run_eval.parse_response.

Two defects motivated these, both found by reading responses rather than by
reading the parser, and both biased in the same direction --- they discarded or
corrupted exactly the evidence the unit accounting needs:

  * confidence was read from the first match anywhere in the response, so a
    model that opens a step with "Confidence: 1 - (upper - lower)/..." was
    recorded at 1.0 when it stated 0.83;
  * an interval whose endpoints carry a unit suffix, ``[224.32K, 253.5K]``,
    did not parse at all --- and a unit suffix is the strongest signal that a
    prediction needs rescaling before it is scored.

The general rule the fix encodes: where a field can match more than once, the
schema block is at the END of a response and everything before it is working.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_eval import parse_response

SCHEMA = (
    "FRAMEWORK: bayesian_inference\n"
    "POINT_ESTIMATE: 0.42\n"
    "CONFIDENCE_INTERVAL: [0.31, 0.53]\n"
    "CONFIDENCE: 0.9"
)


def test_plain_schema_block():
    r = parse_response(SCHEMA)
    assert r["framework"] == "bayesian_inference"
    assert r["point_estimate"] == 0.42
    assert r["confidence_interval"] == (0.31, 0.53)
    assert r["confidence"] == 0.9


def test_confidence_ignores_prose_before_the_schema():
    """The defect: 'Confidence: 1 - (...)' read as a stated confidence of 1.0."""
    text = (
        "Confidence: 1 - (upper bound - lower bound) / (upper + lower)\n"
        "= 1 - (0.152 - 0.049) / (0.152 + 0.049)\n"
        "= 0.83\n\n" + SCHEMA.replace("CONFIDENCE: 0.9", "CONFIDENCE: 0.83")
    )
    assert parse_response(text)["confidence"] == 0.83


def test_last_schema_block_wins():
    """Reasoning models emit a draft block inside <think> and a final one after."""
    text = (
        "<think>\nFRAMEWORK: decision_theory\nPOINT_ESTIMATE: 1.0\n"
        "CONFIDENCE: 0.8\n</think>\n\n"
        "FRAMEWORK: decision_theory\nPOINT_ESTIMATE: 2.0\nCONFIDENCE: 0.5"
    )
    r = parse_response(text)
    assert r["point_estimate"] == 2.0
    assert r["confidence"] == 0.5


def test_framework_takes_the_final_declaration():
    text = "FRAMEWORK: bayesian\nthinking...\nFRAMEWORK: bayesian_inference\nPOINT_ESTIMATE: 1"
    assert parse_response(text)["framework"] == "bayesian_inference"


def test_interval_with_thousands_suffix():
    r = parse_response("POINT_ESTIMATE: 253.5K\nCONFIDENCE_INTERVAL: [224.32K, 253.5K]")
    assert r["confidence_interval"] == (224.32, 253.5)
    assert r["point_estimate"] == 253.5


def test_interval_with_percent_suffix():
    r = parse_response("CONFIDENCE_INTERVAL: [43.5%, 78.5%]")
    assert r["confidence_interval"] == (43.5, 78.5)


def test_interval_with_unit_suffix():
    r = parse_response("CONFIDENCE_INTERVAL: [58.7 µg/m³, 68.0 µg/m³]")
    assert r["confidence_interval"] == (58.7, 68.0)


def test_currency_marks_are_stripped():
    r = parse_response("POINT_ESTIMATE: $44.0\nCONFIDENCE_INTERVAL: [$33.2, $46.1]")
    assert r["point_estimate"] == 44.0
    assert r["confidence_interval"] == (33.2, 46.1)


def test_numbers_are_recorded_as_written():
    """Deciding what unit they are in is unit_accounting's job, not the parser's."""
    r = parse_response("POINT_ESTIMATE: 163195\nCONFIDENCE_INTERVAL: [142820K, 183570K]")
    assert r["point_estimate"] == 163195
    assert r["confidence_interval"] == (142820.0, 183570.0)


def test_negative_and_signed_values():
    r = parse_response("POINT_ESTIMATE: -47.1\nCONFIDENCE_INTERVAL: [-48.4, -33.3]")
    assert r["point_estimate"] == -47.1
    assert r["confidence_interval"] == (-48.4, -33.3)


def test_second_interval_line_supersedes_the_first():
    """A model that corrects itself in the schema block means the correction."""
    text = (
        "CONFIDENCE_INTERVAL: [-0.126, 0.141] is not correct for this estimate, so\n"
        "CONFIDENCE_INTERVAL: [0, 2.5]\nCONFIDENCE: 0.95"
    )
    assert parse_response(text)["confidence_interval"] == (0.0, 2.5)


def test_missing_fields_fall_back():
    r = parse_response("I cannot answer this question.")
    assert r["point_estimate"] is None
    assert r["confidence_interval"] is None
    assert r["framework"] is None
    assert r["confidence"] == 0.5


def test_confidence_default_is_not_treated_as_stated():
    """A response with no confidence field must not read as confident."""
    assert parse_response("POINT_ESTIMATE: 1.0")["confidence"] == 0.5
