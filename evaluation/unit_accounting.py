#!/usr/bin/env python3
"""
unit_accounting.py — the unit the prompt never asked for.

MURU's prompt says "provide your final answer as a single number". Many stems
denominate their quantities in an abbreviated unit --- "the investment cost is
$473K", "the base rate is between 2.00% and 4.20%" --- and the ground truth is
stored in that abbreviated unit (44.0 meaning $44K, 0.254 meaning 25.4%). A
model that answers 44000, or 25.4, has read the problem correctly and answered
in a unit the prompt never ruled out. Scored against a bare interval, it is
marked wrong.

This module measures how much of the panel's recorded error is that, and
nothing else. It is written to be conservative in the one direction that
matters: a rescale is credited only when the model's *own stated interval*
lands on the ground-truth interval under the same factor. Requiring that
corroboration is what separates "answered in dollars throughout" from
"computed a number that happens to be a thousand times too large", and the
distinction is not academic --- on this panel there are answers of both kinds,
and a rule that only checks whether the rescaled point estimate falls inside a
wide ground-truth interval credits the second kind too. Predictions that carry
no usable interval cannot be corroborated and are left scored as errors, so
the corrected accuracy reported here is a lower bound on the correction.

Statuses
  correct                      scored correct as-is
  unit_mismatch                right answer, different admissible unit
                               (point *and* interval agree under one factor)
  unit_mismatch_uncorroborated point rescales into range, interval does not
                               corroborate it --- left as an error, counted
  incorrect                    wrong under every admissible unit

The fix for future runs is in the prompt, not here: ``run_eval.py`` now states
the unit convention explicitly. The archived panel predates that instruction,
which is why the correction has to be applied post hoc and reported as its own
accounting rather than folded silently into the headline numbers.
"""

import re

# Thousands: a stem that writes "$473K" invites an answer in dollars.
THOUSANDS = 1000.0
# Percent: a stem that quotes rates as "4.20%" invites an answer in percent
# where the ground truth is a probability.
PERCENT = 100.0

# How much of the model's rescaled interval must coincide with the
# ground-truth interval (intersection over union) before a rescale is credited.
CORROBORATION_IOU = 0.5

_THOUSANDS_RE = re.compile(r"\d\s*K\b")


def admissible_factors(problem):
    """Unit readings this problem's wording leaves open, largest first.

    Derived from the stem and the ground-truth answer text alone --- never
    from whether a particular rescale would make a particular model right.
    """
    stem = problem.get("stem", "")
    answer = str(problem.get("ground_truth", {}).get("answer", ""))
    factors = []
    if _THOUSANDS_RE.search(stem):
        factors.append(THOUSANDS)
    if "%" in stem or "%" in answer:
        factors.append(PERCENT)
    return factors


def _iou(a, b, c, d):
    lo_a, hi_a = min(a, b), max(a, b)
    lo_b, hi_b = min(c, d), max(c, d)
    inter = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    union = max(hi_a, hi_b) - min(lo_a, lo_b)
    return inter / union if union > 0 else 0.0


def classify_prediction(point, interval, problem):
    """Score one prediction against the ground truth, unit-aware."""
    if point is None:
        return "incorrect"
    lo, hi = problem["ground_truth"]["confidence_interval"]
    if lo <= point <= hi:
        return "correct"

    usable = (
        interval is not None
        and len(interval) == 2
        and all(x is not None for x in interval)
        and interval[0] != interval[1]
    )
    for k in admissible_factors(problem):
        if not (lo <= point / k <= hi):
            continue
        if usable and _iou(interval[0] / k, interval[1] / k, lo, hi) >= CORROBORATION_IOU:
            return "unit_mismatch"
        return "unit_mismatch_uncorroborated"
    return "incorrect"


def credited_factor(point, interval, problem):
    """The unit the model's own answer is denominated in, or 1.0.

    Returns a factor only for a corroborated mismatch, so it can be applied to
    the whole prediction --- point estimate *and* interval --- without
    rescaling anything on the strength of a coincidence. Rescaling the
    interval matters as much as the point: an interval stated in dollars
    against a ground truth in thousands is a thousand times too wide, which
    reads as extreme hedging in any width or interval-score statistic.
    """
    if classify_prediction(point, interval, problem) != "unit_mismatch":
        return 1.0
    lo, hi = problem["ground_truth"]["confidence_interval"]
    for k in admissible_factors(problem):
        if lo <= point / k <= hi:
            return k
    return 1.0


def is_correct(status, unit_aware):
    """Correctness under one of the two accountings."""
    if unit_aware:
        return status in ("correct", "unit_mismatch")
    return status == "correct"


def report(statuses):
    """Counts and the accuracy each accounting produces."""
    n = len(statuses)
    if not n:
        return {"n": 0}
    counts = {
        s: sum(1 for x in statuses if x == s)
        for s in (
            "correct",
            "unit_mismatch",
            "unit_mismatch_uncorroborated",
            "incorrect",
        )
    }
    return {
        "n": n,
        "counts": counts,
        "accuracy_raw": counts["correct"] / n,
        "accuracy_unit_aware": (counts["correct"] + counts["unit_mismatch"]) / n,
        "accuracy_upper_bound": (
            n - counts["incorrect"]
        ) / n,
        "unit_mismatch_share_of_errors": (
            counts["unit_mismatch"] / (n - counts["correct"])
            if n > counts["correct"]
            else 0.0
        ),
    }
