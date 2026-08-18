"""Tests for evaluation/defect_leaveout.py.

The leave-out exists to answer one question — how much of the published panel
result rests on the 29 test items carrying an item-construction defect — and it
is only worth anything if its "all items" column is the *same* computation as
the leaderboard's. So the load-bearing test here is the consistency one: every
all-items cell must reproduce `real_llm_summary.json` to the last decimal. If
that drifts, the leave-out is comparing two different things and the delta it
reports is meaningless.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluation import defect_leaveout as dl

SUMMARY_PATH = PROJECT_ROOT / "evaluation" / "baselines" / "real_llm_summary.json"


@pytest.fixture(scope="module")
def result():
    return dl.analyse()


@pytest.fixture(scope="module")
def summary():
    if not SUMMARY_PATH.exists():
        pytest.skip("real_llm_summary.json not built; run `make reanalyze`")
    with open(SUMMARY_PATH) as handle:
        return json.load(handle)


def test_defective_ids_flags_a_known_broken_item():
    problems = dl.load_problems(PROJECT_ROOT / "data" / "test")
    flagged = dl.defective_ids(problems)
    # MURU-1258 is the 482.3 mmHg stem; MURU-2384 the 0.001-wide interval.
    assert "MURU-1258" in flagged
    assert "MURU-2384" in flagged


def test_defective_ids_leaves_the_bulk_of_the_split_alone():
    problems = dl.load_problems(PROJECT_ROOT / "data" / "test")
    flagged = dl.defective_ids(problems)
    assert 0 < len(flagged) < len(problems) // 4, len(flagged)


def test_clean_and_defective_partition_the_answered_set(result):
    for row in result["rows"]:
        assert row["clean"]["n"] + row["defective"]["n"] == row["all"]["n"], row["model"]


def test_all_items_column_reproduces_the_published_leaderboard(result, summary):
    """The leave-out must not be a second, subtly different scoring path."""
    by_display = {
        v["display"]: v for k, v in summary.items() if isinstance(v, dict) and "display" in v
    }
    assert by_display, "no model rows in real_llm_summary.json"
    checked = 0
    for row in result["rows"]:
        expected = by_display.get(row["model"])
        if expected is None:
            continue
        checked += 1
        assert row["all"]["n"] == expected["n_evaluated"], row["model"]
        assert row["all"]["accuracy"]["point"] == pytest.approx(
            expected["metrics"]["accuracy"]["point"], abs=1e-9
        ), row["model"]
        assert row["all"]["ece"]["point"] == pytest.approx(
            expected["metrics"]["ece"]["point"], abs=1e-9
        ), row["model"]
    assert checked >= 5, f"only cross-checked {checked} models"


def test_leaveout_does_not_reorder_the_leaderboard(result):
    """If it did, the paper could not keep reporting the all-items ordering."""
    assert result["ranking_preserved"], result["ranking"]


def test_headline_null_survives_the_leaveout(result):
    """The finding is that accuracy and calibration-in-level are separable.

    The pre-correction claim rested on rho = -0.90. Both accountings here have
    to stay far away from that, or the null is an artefact of the broken items.
    """
    for which, rho in result["accuracy_ece_spearman"].items():
        assert abs(rho) < 0.5, f"{which}: rho = {rho:+.2f}"


def test_score_is_the_plain_arithmetic_it_claims_to_be():
    problems = dl.load_problems(PROJECT_ROOT / "data" / "test")[:20]
    preds = []
    from evaluation.metrics import Prediction

    for i, problem in enumerate(problems):
        gt = problem["ground_truth"]
        # Alternate exactly-right and far-wrong answers.
        value = gt["point_estimate"] if i % 2 == 0 else gt["point_estimate"] + 1e6
        preds.append(
            Prediction(
                problem_id=problem["id"],
                predicted_answer=value,
                predicted_confidence=0.8,
                predicted_interval=tuple(gt["confidence_interval"]),
                predicted_framework=problem["required_framework"],
                raw_response="",
            )
        )
    scored = dl.score(problems, preds)
    assert scored["n"] == 20
    assert scored["accuracy"]["point"] == pytest.approx(0.5)
    lo, hi = scored["accuracy"]["ci95"]
    assert lo <= 0.5 <= hi


def test_score_returns_none_on_an_empty_subset():
    assert dl.score([], []) is None
