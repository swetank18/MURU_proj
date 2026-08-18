"""Tests for scripts/audit_item_defects.py and the generator fixes behind it.

The three item-construction defects were found by reading 86 sampled model
errors, which only surfaces a defect a model happened to trip over — the
diastolic-blood-pressure stems were caught because one model flagged 482.3 mmHg
as a typo, and the corpus turned out to hold implausible fuel-consumption and
per-hectare-yield figures nobody had queried. So the tests here come in two
halves: the checks must fire on the exact pre-errata wording (kept verbatim
below), and the fixed generator must emit nothing the checks fire on.

As with the failure codebook, most of the check tests are negative cases. An
audit that over-fires would push a clean item onto an errata list and, worse,
make the next real defect look like noise.
"""

import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_item_defects as audit
import generate_problems as gp


def problem(author, stem="", point_estimate=0.5, ci=(0.4, 0.6)):
    return {
        "id": "MURU-9999",
        "stem": stem,
        "ground_truth": {
            "point_estimate": point_estimate,
            "confidence_interval": list(ci),
        },
        "metadata": {"author": author},
    }


# ──────────────────────────────────────────────────────────────
# D1 — physically implausible stem values
# ──────────────────────────────────────────────────────────────

# Verbatim from MURU-1258, the worst of the pre-errata blood-pressure stems.
BP_IMPOSSIBLE = (
    "A clinical study study measures the diastolic blood pressure (mmHg) for a "
    "sample of 30 patients. The sample mean is 482.3 mmHg with a sample standard "
    "deviation of 91.6 mmHg. Assuming the data are approximately normally "
    "distributed, estimate the true population mean and provide a 95% confidence "
    "interval."
)

BP_PLAUSIBLE = BP_IMPOSSIBLE.replace("482.3 mmHg", "78.4 mmHg").replace("91.6 mmHg", "9.1 mmHg")


def test_d1_fires_on_impossible_blood_pressure():
    findings = audit.check_d1(problem("generator_sample_mean", BP_IMPOSSIBLE))
    assert len(findings) == 1
    assert "482.3" in findings[0] and findings[0].startswith("D1")


def test_d1_silent_on_plausible_blood_pressure():
    assert audit.check_d1(problem("generator_sample_mean", BP_PLAUSIBLE)) == []


def test_d1_accepts_the_upper_edge_of_the_plausible_range():
    """130 mmHg is a hypertensive crisis, not an impossibility. It must pass."""
    stem = BP_IMPOSSIBLE.replace("482.3 mmHg", "130.0 mmHg")
    assert audit.check_d1(problem("generator_sample_mean", stem)) == []


def test_d1_catches_quantities_no_model_ever_queried():
    """The corpus held 18 fuel-consumption and 16 yield stems as bad as the BP ones."""
    stem = (
        "A automotive testing study measures the fuel consumption (L/100km) for a "
        "sample of 25 vehicles. The sample mean is 424.1 L/100km with a sample "
        "standard deviation of 84.0 L/100km. Assuming the data are approximately "
        "normally distributed, estimate the true population mean and provide a 95% "
        "confidence interval."
    )
    findings = audit.check_d1(problem("generator_sample_mean", stem))
    assert len(findings) == 1 and "424.1" in findings[0]


def test_d1_ignores_other_templates():
    assert audit.check_d1(problem("generator_simpsons_paradox", BP_IMPOSSIBLE)) == []


def test_d1_reports_an_unregistered_measurement_rather_than_passing_it():
    stem = BP_IMPOSSIBLE.replace("diastolic blood pressure (mmHg)", "cromulence (furlongs)")
    findings = audit.check_d1(problem("generator_sample_mean", stem))
    assert len(findings) == 1 and "no plausible range" in findings[0]


# ──────────────────────────────────────────────────────────────
# D2 — contradictory or ambiguous test accuracy
# ──────────────────────────────────────────────────────────────

# Verbatim from MURU-0296: the stem says 92%, the colleague says 95%, and the
# ground truth silently uses 95% as the sensitivity.
TRAP_LEGACY = (
    "A university's plagiarism detection software flags plagiarized work with 92% "
    "accuracy. The test also has a specificity of 88% (correctly identifying "
    "negatives). A colleague presents this data and concludes: 'Since the test is "
    "95% accurate, when it flags someone, there's a 95% chance they're actually "
    "plagiarized.' The base rate of submissions that are actually plagiarized in "
    "this student submissions pool is estimated between 1.60% and 5.60%. Is your "
    "colleague's reasoning correct? What is the actual probability that a flagged "
    "student submission is truly plagiarized? How does the uncertain base rate "
    "affect your answer?"
)

TRAP_FIXED = (
    "A university's plagiarism detection software correctly flags 92% of the work "
    "that is genuinely plagiarized. The test also has a specificity of 88% "
    "(correctly identifying negatives). A colleague presents this data and "
    "concludes: 'Since the test is 92% accurate, when it flags someone, there's a "
    "92% chance they're actually plagiarized.' The base rate of submissions that "
    "are actually plagiarized in this student submissions pool is estimated "
    "between 1.60% and 5.60%. Is your colleague's reasoning correct? What is the "
    "actual probability that a flagged student submission is truly plagiarized? "
    "How does the uncertain base rate affect your answer?"
)


def test_d2_fires_on_the_stem_quote_mismatch():
    findings = audit.check_d2(problem("generator_base_rate_trap", TRAP_LEGACY))
    assert any("stem states 92% but the colleague's quote uses 95%" in f for f in findings)


def test_d2_fires_on_the_unsolvable_overall_accuracy_reading():
    findings = audit.check_d2(problem("generator_base_rate_trap", TRAP_LEGACY))
    assert any("implies sensitivity" in f for f in findings)


def test_d2_silent_on_the_fixed_wording():
    assert audit.check_d2(problem("generator_base_rate_trap", TRAP_FIXED)) == []


def test_d2_still_fires_if_the_fixed_wording_disagrees_with_the_quote():
    """The explicit-sensitivity phrasing must not mask a numeric contradiction."""
    stem = TRAP_FIXED.replace("Since the test is 92% accurate", "Since the test is 95% accurate")
    findings = audit.check_d2(problem("generator_base_rate_trap", stem))
    assert len(findings) == 1 and "92%" in findings[0] and "95%" in findings[0]


def test_d2_allows_a_bare_accuracy_stem_that_is_still_solvable():
    """Ambiguity only bites where the overall-accuracy reading implies sens > 1.

    At a 35% base rate the reading is merely a second admissible one, not an
    impossibility, so the audit must stay silent rather than flag every legacy
    stem on sight.
    """
    stem = TRAP_LEGACY.replace(
        "Since the test is 95% accurate", "Since the test is 92% accurate"
    ).replace("between 1.60% and 5.60%", "between 33.00% and 37.00%")
    assert audit.check_d2(problem("generator_base_rate_trap", stem)) == []


def test_d2_ignores_other_templates():
    assert audit.check_d2(problem("generator_simpsons_paradox", TRAP_LEGACY)) == []


# ──────────────────────────────────────────────────────────────
# D3 — ground-truth interval narrower than the invited precision
# ──────────────────────────────────────────────────────────────

def test_d3_fires_on_the_simpsons_interval_that_started_this():
    """MURU-2384: 0.001 wide, so a correct 0.0579 rounded to 0.058 falls outside."""
    findings = audit.check_d3(problem("generator_simpsons_paradox", point_estimate=0.057, ci=(0.056, 0.057)))
    assert len(findings) == 1 and findings[0].startswith("D3")


def test_d3_silent_at_exactly_two_units_of_the_last_decimal():
    assert audit.check_d3(problem("x", point_estimate=0.057, ci=(0.056, 0.058))) == []


def test_d3_scales_the_floor_to_the_published_precision():
    """A four-decimal reliability may legitimately sit inside a 0.0008 interval."""
    assert audit.check_d3(problem("x", point_estimate=0.9988, ci=(0.9984, 0.9992))) == []
    assert audit.check_d3(problem("x", point_estimate=0.9988, ci=(0.9988, 0.9989))) != []


def test_d3_handles_integer_valued_bounds():
    assert audit.check_d3(problem("x", point_estimate=58, ci=(57, 59))) == []


def test_d3_ignores_problems_with_no_interval():
    p = problem("x")
    p["ground_truth"]["confidence_interval"] = None
    assert audit.check_d3(p) == []


def test_decimals_counts_published_precision():
    assert audit.decimals(0.057) == 3
    assert audit.decimals(0.0570) == 3
    assert audit.decimals(58.0) == 0
    assert audit.decimals(0.9984) == 4


# ──────────────────────────────────────────────────────────────
# The generator must not reintroduce any of them
# ──────────────────────────────────────────────────────────────

DEFECT_PRONE_TEMPLATES = [
    "sample_mean",
    "base_rate_trap",
    "simpsons_paradox",
    "hierarchical_bayes",
    "parallel_redundancy",
]


@pytest.mark.parametrize("template_name", DEFECT_PRONE_TEMPLATES)
def test_generator_emits_no_defects(template_name):
    """Each of these templates produced defects before the errata fix."""
    template = gp.TEMPLATES[template_name]
    low, high = template.difficulty_range
    rng = random.Random(20260818)
    state = random.getstate()
    random.seed(rng.randrange(2**32))
    try:
        for i in range(250):
            item = template.generate(9000 + i, rng.randint(low, high))
            found = [m for check in audit.CHECKS for m in check(item)]
            assert not found, f"{template_name} {item['id']}: {found}"
    finally:
        random.setstate(state)


def test_sample_mean_respects_the_physical_range_of_every_quantity():
    """Every context must declare a range, and it must sit inside the audit's."""
    for ctx in gp.SAMPLE_MEAN_CONTEXTS:
        low, high = ctx["mean_range"]
        assert low < high
        plausible = audit.PLAUSIBLE_MEAN_RANGE[ctx["measurement"]]
        assert plausible[0] <= low and high <= plausible[1], ctx["measurement"]


def test_sample_mean_stems_do_not_stutter():
    """'A clinical study study measures ...' shipped in 24 items."""
    state = random.getstate()
    random.seed(4242)
    try:
        for i in range(200):
            item = gp.TEMPLATES["sample_mean"].generate(9000 + i, random.randint(1, 5))
            assert "study study" not in item["stem"]
    finally:
        random.setstate(state)


def test_base_rate_trap_states_one_figure_and_states_it_as_sensitivity():
    state = random.getstate()
    random.seed(99)
    try:
        for i in range(200):
            item = gp.TEMPLATES["base_rate_trap"].generate(9000 + i, random.randint(2, 4))
            stem = item["stem"]
            explicit = audit.TRAP_SENSITIVITY_RE.search(stem)
            quoted = audit.TRAP_QUOTE_ACC_RE.search(stem)
            assert explicit, stem
            assert quoted and explicit.group(1) == quoted.group(1), stem
    finally:
        random.setstate(state)


def test_simpsons_intervals_clear_the_declared_floor():
    state = random.getstate()
    random.seed(1234)
    try:
        for i in range(200):
            item = gp.TEMPLATES["simpsons_paradox"].generate(9000 + i, random.randint(3, 5))
            low, high = item["ground_truth"]["confidence_interval"]
            assert round(high - low, 4) >= gp.MIN_SIMPSONS_CI_WIDTH, item["stem"]
    finally:
        random.setstate(state)


def test_simpsons_still_produces_the_paradox_it_is_named_for():
    """The width floor must not be bought by dropping the reversal."""
    state = random.getstate()
    random.seed(555)
    try:
        for i in range(100):
            item = gp.TEMPLATES["simpsons_paradox"].generate(9000 + i, 4)
            assert "Simpson's Paradox" in item["ground_truth"]["answer"]
            assert item["ground_truth"]["point_estimate"] > 0, "X must stay better once adjusted"
    finally:
        random.setstate(state)
