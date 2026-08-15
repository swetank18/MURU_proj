#!/usr/bin/env python3
"""
calibration_metrics.py — Proper scoring rules and hardened calibration metrics.

Accuracy@CI and ECE alone are both gameable, in opposite directions:

  * Accuracy@CI rewards a wide model interval and does not look at its width
    at all, so a model that answers "somewhere between minus and plus
    infinity" scores perfectly. A *proper* scoring rule closes this: the
    interval (Winkler) score charges for width and for missing the target in
    the same number, so hedging and overconfidence trade off against each
    other and cannot both be optimised away.
  * ECE is binning-dependent and biased upward at finite n — at n = 301 a
    perfectly calibrated model still scores ECE > 0. Reporting a single
    10-equal-width-bin number hides both facts, so this module reports the
    binning sensitivity and a bias-corrected estimate alongside.

Everything here is stdlib-only and deterministic given a seed, so it runs on
the committed archives with no external dependency.

Metrics implemented
  interval_score            Winkler score for a (1-alpha) interval
  relative_width            model interval width / ground-truth width
  ece                       equal-width or equal-mass binning, any bin count
  ece_binning_sensitivity   spread across schemes x bin counts
  ece_debiased              observed ECE minus its finite-n null floor
  brier_decomposition       Brier score, Murphy reliability/resolution/uncertainty
  auroc                     discrimination of stated confidence over correctness
"""

import random
from collections import defaultdict

# ── Interval scoring ───────────────────────────────────────────────────


def interval_score(lower, upper, y, alpha=0.1):
    """Winkler interval score for a central (1-alpha) interval.

    IS = (u - l) + (2/alpha)(l - y) if y < l
                 + (2/alpha)(y - u) if y > u

    Lower is better. Proper: minimised in expectation by the true
    (1-alpha) predictive interval, so it cannot be gamed by widening.
    """
    if upper < lower:
        lower, upper = upper, lower
    score = upper - lower
    if y < lower:
        score += (2.0 / alpha) * (lower - y)
    elif y > upper:
        score += (2.0 / alpha) * (y - upper)
    return score


def normalized_interval_score(lower, upper, y, gt_width, alpha=0.1):
    """Interval score in units of the ground-truth interval width.

    MURU problems span many orders of magnitude (ground-truth widths run from
    1e-3 to ~1e3), so raw interval scores are dominated by the largest-scale
    problems and cannot be averaged. Dividing by the defensible-answer width
    makes the score unitless and comparable across the split: 1.0 means "as
    costly as emitting exactly the ground-truth interval and hitting it".
    """
    if not gt_width:
        return None
    return interval_score(lower, upper, y, alpha) / gt_width


def relative_width(lower, upper, gt_width):
    """Model interval width as a multiple of the ground-truth width."""
    if not gt_width:
        return None
    return abs(upper - lower) / gt_width


# ── ECE family ─────────────────────────────────────────────────────────


def _equal_width_bins(conf, n_bins):
    bins = defaultdict(list)
    for i, c in enumerate(conf):
        bins[min(int(c * n_bins), n_bins - 1)].append(i)
    return list(bins.values())


def _equal_mass_bins(conf, n_bins):
    """Adaptive binning: equal counts per bin, so no bin is empty or dominant.

    Ties in confidence are kept in the same bin — verbalized confidence is
    heavily tied (models say 0.9 constantly), and splitting a tied block
    across bins would manufacture calibration signal that is not there.
    """
    order = sorted(range(len(conf)), key=lambda i: conf[i])
    if not order:
        return []
    target = max(1, len(order) // n_bins)
    bins, current = [], []
    for pos, i in enumerate(order):
        current.append(i)
        at_capacity = len(current) >= target
        boundary_is_tie = pos + 1 < len(order) and conf[order[pos + 1]] == conf[i]
        if at_capacity and not boundary_is_tie and len(bins) < n_bins - 1:
            bins.append(current)
            current = []
    if current:
        bins.append(current)
    return bins


def ece(correct, conf, n_bins=10, scheme="equal_width"):
    """Expected Calibration Error under a chosen binning scheme."""
    if not correct:
        return 0.0
    groups = (
        _equal_mass_bins(conf, n_bins)
        if scheme == "equal_mass"
        else _equal_width_bins(conf, n_bins)
    )
    n = len(correct)
    total = 0.0
    for idxs in groups:
        if not idxs:
            continue
        acc = sum(correct[i] for i in idxs) / len(idxs)
        avg_conf = sum(conf[i] for i in idxs) / len(idxs)
        total += (len(idxs) / n) * abs(acc - avg_conf)
    return total


def ece_binning_sensitivity(correct, conf, bin_counts=(5, 10, 15, 20)):
    """ECE across both schemes and several bin counts.

    The spread is the honest error bar on a binned ECE: if it is wide, the
    headline ECE is a property of the binning as much as of the model.
    """
    grid = {}
    for scheme in ("equal_width", "equal_mass"):
        for k in bin_counts:
            grid[f"{scheme}_{k}"] = ece(correct, conf, n_bins=k, scheme=scheme)
    values = list(grid.values())
    return {
        "grid": grid,
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def ece_debiased(correct, conf, n_bins=10, scheme="equal_width", n_sim=2000, seed=0xCA11B):
    """ECE with its finite-sample floor subtracted.

    A perfectly calibrated model still scores ECE > 0 at finite n, because
    each bin's empirical accuracy scatters around its confidence. The floor is
    estimated by simulating correctness from the model's own stated
    confidences (the null of perfect calibration) and taking the mean ECE of
    those draws. Reported value is max(0, observed - floor).
    """
    if not correct:
        return {"observed": 0.0, "null_floor": 0.0, "debiased": 0.0}
    observed = ece(correct, conf, n_bins=n_bins, scheme=scheme)
    rng = random.Random(seed)
    floor_total = 0.0
    for _ in range(n_sim):
        sim = [1 if rng.random() < c else 0 for c in conf]
        floor_total += ece(sim, conf, n_bins=n_bins, scheme=scheme)
    floor = floor_total / n_sim
    return {
        "observed": observed,
        "null_floor": floor,
        "debiased": max(0.0, observed - floor),
    }


# ── Brier score and Murphy decomposition ───────────────────────────────


def brier_decomposition(correct, conf, n_bins=10):
    """Brier score with Murphy's reliability / resolution / uncertainty split.

    BS = REL - RES + UNC, where REL is miscalibration (lower better), RES is
    how far bin accuracies move away from the base rate (higher better), and
    UNC is the irreducible variance of the outcome. ECE conflates a model that
    is badly calibrated with one that is merely uninformative; this separates
    them.
    """
    if not correct:
        return None
    n = len(correct)
    brier = sum((c - y) ** 2 for c, y in zip(conf, correct)) / n
    base = sum(correct) / n
    rel = res = 0.0
    for idxs in _equal_mass_bins(conf, n_bins):
        if not idxs:
            continue
        w = len(idxs) / n
        acc = sum(correct[i] for i in idxs) / len(idxs)
        avg_conf = sum(conf[i] for i in idxs) / len(idxs)
        rel += w * (avg_conf - acc) ** 2
        res += w * (acc - base) ** 2
    return {
        "brier": brier,
        "reliability": rel,
        "resolution": res,
        "uncertainty": base * (1 - base),
    }


# ── Discrimination ─────────────────────────────────────────────────────


def auroc(correct, conf):
    """AUROC of stated confidence as a predictor of correctness.

    Computed as the tie-corrected Mann-Whitney statistic. A model can be badly
    calibrated in level yet still rank its own correct answers above its wrong
    ones (AUROC high, ECE high) — which is exactly the property that matters
    for selective prediction, and which ECE alone cannot see.
    """
    pos = [c for c, y in zip(conf, correct) if y]
    neg = [c for c, y in zip(conf, correct) if not y]
    if not pos or not neg:
        return None
    order = sorted(range(len(conf)), key=lambda i: conf[i])
    ranks = [0.0] * len(conf)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and conf[order[j + 1]] == conf[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed midrank over the tied block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i, y in enumerate(correct) if y)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


# ── Aggregate over a model's per-problem records ───────────────────────


HEDGE_MULTIPLE = 10.0  # a model interval this many times the defensible width


def sharpness_summary(records):
    """Interval-score / width statistics over one model's answered problems.

    ``records`` are dicts with keys: lower, upper, gt_point, gt_width, alpha.
    Entries whose model gave no interval are counted but excluded from the
    width and score statistics (reported as ``interval_rate``).

    Report the median and the upper tail, never the mean alone: the relative
    width distribution is extremely heavy-tailed (a model that answers [0, 1]
    on a problem whose defensible interval is 0.001 wide scores 1000x), so the
    mean is set by a handful of catastrophic hedges and says nothing about
    typical behaviour. ``hedge_rate`` counts that tail directly.
    """
    nis, rel_w, hits = [], [], []
    n_with_interval = 0
    for r in records:
        if r.get("lower") is None or r.get("upper") is None:
            continue
        n_with_interval += 1
        lo, hi = sorted((r["lower"], r["upper"]))
        s = normalized_interval_score(lo, hi, r["gt_point"], r["gt_width"], r["alpha"])
        w = relative_width(lo, hi, r["gt_width"])
        if s is not None:
            nis.append(s)
        if w is not None:
            rel_w.append(w)
        hits.append(int(lo <= r["gt_point"] <= hi))

    def quantile(xs, q):
        if not xs:
            return None
        s = sorted(xs)
        return s[min(int(q * (len(s) - 1)), len(s) - 1)]

    return {
        "n_records": len(records),
        "n_with_interval": n_with_interval,
        "interval_rate": n_with_interval / len(records) if records else None,
        "mean_normalized_interval_score": (sum(nis) / len(nis)) if nis else None,
        "median_normalized_interval_score": quantile(nis, 0.5),
        "p90_normalized_interval_score": quantile(nis, 0.9),
        "mean_relative_width": (sum(rel_w) / len(rel_w)) if rel_w else None,
        "median_relative_width": quantile(rel_w, 0.5),
        "p90_relative_width": quantile(rel_w, 0.9),
        "hedge_rate": (
            sum(1 for w in rel_w if w > HEDGE_MULTIPLE) / len(rel_w) if rel_w else None
        ),
        "pe_coverage": (sum(hits) / len(hits)) if hits else None,
    }


def full_summary(correct, conf, records=None, n_bins=10):
    """Every hardened metric for one model, in one dict."""
    out = {
        "ece_equal_width_10": ece(correct, conf, 10, "equal_width"),
        "ece_equal_mass_10": ece(correct, conf, 10, "equal_mass"),
        "ece_sensitivity": ece_binning_sensitivity(correct, conf),
        "ece_debiased": ece_debiased(correct, conf, n_bins=n_bins),
        "brier": brier_decomposition(correct, conf, n_bins=n_bins),
        "auroc": auroc(correct, conf),
    }
    if records is not None:
        out["sharpness"] = sharpness_summary(records)
    return out
