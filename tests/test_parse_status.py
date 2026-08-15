"""Tests for evaluation/parse_status.py.

The point of the module is that a response the model got wrong, a response the
harness could not read, and a request that never returned are three different
events. These tests pin that separation down, because collapsing any two of
them silently changes what the leaderboard means.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.parse_status import (
    MODEL_FAILURE_STATUSES,
    PROVIDER_FAILURE_STATUSES,
    classify_entry,
    status_report,
)


def entry(pid="MURU-0001", response="", parsed=None, error=None):
    e = {"problem_id": pid, "response": response, "parsed": parsed or {}}
    if error is not None:
        e["error"] = error
    return e


def test_parsed_answer_is_ok():
    assert classify_entry(entry(parsed={"point_estimate": 0.42})) == "ok"


def test_endpoint_404_is_provider_failure():
    e = entry(error="Error code: 404 - model `qwen/qwen3-32b` does not exist")
    assert classify_entry(e) == "endpoint_unavailable"
    assert "endpoint_unavailable" in PROVIDER_FAILURE_STATUSES


def test_rate_limit_and_timeout_are_distinguished():
    assert classify_entry(entry(error="429 rate_limit_exceeded")) == "rate_limited"
    assert classify_entry(entry(error="Request timed out")) == "timeout"


def test_truncated_response_mid_sentence():
    e = entry(response="We compute the posterior by first noting that the")
    assert classify_entry(e) == "truncated"


def test_truncated_response_mid_number():
    # Ends on a period, but it is a sliced decimal, not a finished sentence.
    e = entry(response="= 0.52 + 1.96 * sqrt(0.1248)\n= 0.52 + 1.96 * 0.")
    assert classify_entry(e) == "truncated"


def test_empty_point_estimate_field():
    e = entry(
        response="FRAMEWORK: bayesian_inference\nPOINT_ESTIMATE: \nCONFIDENCE: 0.5",
        parsed={"framework": "bayesian_inference", "confidence": 0.5},
    )
    assert classify_entry(e) == "missing_field"


def test_off_schema_but_answered():
    e = entry(response="**Point Estimate:** 0.65\n\nThat concludes the analysis.")
    assert classify_entry(e) == "format_variant"
    assert "format_variant" in MODEL_FAILURE_STATUSES


def test_complete_response_without_schema():
    e = entry(response="The answer depends on assumptions we cannot pin down here.")
    assert classify_entry(e) == "no_schema"


def test_refusal():
    e = entry(response="I cannot provide a numerical answer to this problem.")
    assert classify_entry(e) == "refused"


def test_status_report_separates_the_two_denominators():
    archive = {
        "raw_results": [
            entry("MURU-0001", parsed={"point_estimate": 1.0}),
            entry("MURU-0002", parsed={"point_estimate": 2.0}),
            entry("MURU-0003", response="the model stopped mid"),          # model
            entry("MURU-0004", error="404 model does not exist"),          # provider
        ]
    }
    ids = {"MURU-0001", "MURU-0002", "MURU-0003", "MURU-0004", "MURU-0005"}
    r = status_report(archive, n_test=len(ids), test_ids=ids)

    # Parse rate is over what the model was actually asked and answered;
    # coverage is over the split. They must not be the same denominator.
    assert r["n_parsed"] == 2
    assert r["n_attempted"] == 3          # excludes the 404
    assert r["parse_rate"] == 2 / 3
    assert r["coverage"] == 3 / 5
    assert r["n_provider_failure"] == 2   # the 404 plus the never-attempted item
    assert r["per_problem"]["MURU-0005"] == "unattempted"
