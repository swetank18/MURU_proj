"""Tests for evaluation/judgment_coding.py.

The coding file is the one artifact in this repository that cannot be
regenerated -- it is the record of somebody reading. So the tests here are
about the apparatus that keeps it honest: the structural validator that
refuses a code without evidence, and the agreement statistic that must not
quietly report a number when there is nothing to compare against.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.judgment_coding import (
    cohens_kappa,
    coded_map,
    confusion,
    group_of,
    validate,
    wilson,
)


def sample(*keys):
    return [
        {"model_slug": k.split(":")[0], "problem_id": k.split(":")[1]} for k in keys
    ]


def coding(*pairs, evidence="because the response says so"):
    return {
        "coder": "t",
        "codes": [
            {"key": k, "primary": c, "secondary": [], "evidence": evidence}
            for k, c in pairs
        ],
    }


# --- the validator ----------------------------------------------------------


def test_clean_coding_validates():
    s = sample("m:MURU-0001", "m:MURU-0002")
    assert validate(coding(("m:MURU-0001", "C3"), ("m:MURU-0002", "R1")), s) == []


def test_a_code_without_evidence_is_rejected():
    """A coder who cannot quote the evidence has not coded the item."""
    s = sample("m:MURU-0001")
    problems = validate(coding(("m:MURU-0001", "C3"), evidence="  "), s)
    assert any("no evidence" in p for p in problems)


def test_XX_may_omit_evidence():
    """XX means there is nothing to quote -- that is what it records."""
    s = sample("m:MURU-0001")
    assert validate(coding(("m:MURU-0001", "XX"), evidence=""), s) == []


def test_unknown_code_is_rejected():
    s = sample("m:MURU-0001")
    problems = validate(coding(("m:MURU-0001", "ZZ")), s)
    assert any("not a code" in p for p in problems)


def test_item_defect_codes_are_valid():
    s = sample("m:MURU-0001", "m:MURU-0002")
    assert validate(coding(("m:MURU-0001", "T1"), ("m:MURU-0002", "D1")), s) == []


def test_uncoded_items_are_reported():
    s = sample("m:MURU-0001", "m:MURU-0002")
    problems = validate(coding(("m:MURU-0001", "C3")), s)
    assert any("uncoded" in p for p in problems)


def test_duplicate_and_foreign_keys_are_reported():
    s = sample("m:MURU-0001")
    c = coding(("m:MURU-0001", "C3"), ("m:MURU-0001", "R1"), ("m:MURU-9999", "C3"))
    problems = validate(c, s)
    assert any("coded twice" in p for p in problems)
    assert any("not in the sample" in p for p in problems)


# --- agreement --------------------------------------------------------------


def test_kappa_is_one_on_perfect_agreement():
    a = {"x": "C3", "y": "R1", "z": "C1"}
    k, n, po, _ = cohens_kappa(a, dict(a))
    assert n == 3 and po == 1.0 and abs(k - 1.0) < 1e-9


def test_kappa_is_zero_at_chance():
    """Two coders using the same marginals but agreeing only by chance."""
    a = {"1": "A", "2": "A", "3": "B", "4": "B"}
    b = {"1": "A", "2": "B", "3": "A", "4": "B"}
    k, _, po, pe = cohens_kappa(a, b)
    assert po == 0.5 and pe == 0.5 and abs(k) < 1e-9


def test_kappa_is_negative_when_worse_than_chance():
    a = {"1": "A", "2": "A", "3": "B", "4": "B"}
    b = {"1": "B", "2": "B", "3": "A", "4": "A"}
    k, *_ = cohens_kappa(a, b)
    assert k < 0


def test_kappa_undefined_when_both_coders_use_one_code():
    """Perfect agreement on a constant is not evidence of reliability."""
    a = {"1": "C3", "2": "C3"}
    k, n, po, pe = cohens_kappa(a, dict(a))
    assert k is None and po == 1.0 and pe == 1.0


def test_kappa_uses_only_shared_items():
    a = {"1": "C3", "2": "R1", "3": "C1"}
    b = {"1": "C3", "2": "R1"}
    _, n, po, _ = cohens_kappa(a, b)
    assert n == 2 and po == 1.0


def test_kappa_on_no_overlap_returns_nothing():
    k, n, po, pe = cohens_kappa({"1": "C3"}, {"2": "C3"})
    assert k is None and n == 0 and po is None and pe is None


def test_confusion_counts_disagreements():
    a = {"1": "C3", "2": "C3"}
    b = {"1": "C3", "2": "R1"}
    assert confusion(a, b)[("C3", "R1")] == 1


# --- reporting helpers ------------------------------------------------------


def test_group_of_separates_item_defects_from_model_failures():
    assert group_of("T1") == "item defect"
    assert group_of("D1") == "item defect"
    assert group_of("C3") == "computation"
    assert group_of("R1") == "reporting"
    assert group_of("R5") == "reporting"
    assert group_of("A1") == "no computation"


def test_wilson_brackets_the_point_estimate():
    lo, hi = wilson(8, 25)
    assert lo < 8 / 25 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_stays_in_range_at_the_boundary():
    assert wilson(0, 25)[0] == 0.0
    assert wilson(25, 25)[1] == 1.0


def test_coded_map_keys_on_item():
    c = coding(("m:MURU-0001", "C3"))
    assert coded_map(c) == {"m:MURU-0001": "C3"}
