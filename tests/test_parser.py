"""Tests for evaluation/run_eval.py response parsing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_eval import parse_response


def test_parse_well_formed_response():
    response = """My reasoning shows that...

FRAMEWORK: bayesian_inference
POINT_ESTIMATE: 0.42
CONFIDENCE_INTERVAL: [0.35, 0.50]
CONFIDENCE: 0.85"""
    result = parse_response(response)
    assert result["framework"] == "bayesian_inference"
    assert result["point_estimate"] == 0.42
    assert result["confidence_interval"] == (0.35, 0.50)
    assert result["confidence"] == 0.85


def test_parse_negative_estimate():
    response = "POINT_ESTIMATE: -3.14\nCONFIDENCE: 0.5"
    result = parse_response(response)
    assert result["point_estimate"] == -3.14


def test_parse_case_insensitive():
    response = "framework: monte_carlo\npoint_estimate: 1.5\nconfidence: 0.7"
    result = parse_response(response)
    assert result["framework"] == "monte_carlo"
    assert result["point_estimate"] == 1.5
    assert result["confidence"] == 0.7


def test_parse_missing_fields_returns_defaults():
    response = "Just some text without any markers."
    result = parse_response(response)
    assert result["framework"] is None
    assert result["point_estimate"] is None
    assert result["confidence_interval"] is None
    assert result["confidence"] == 0.5  # documented default


def test_parse_interval_with_extra_spaces():
    response = "CONFIDENCE_INTERVAL: [  0.1  ,  0.9  ]"
    result = parse_response(response)
    assert result["confidence_interval"] == (0.1, 0.9)


def test_parse_integer_estimate():
    response = "POINT_ESTIMATE: 42"
    result = parse_response(response)
    assert result["point_estimate"] == 42.0
