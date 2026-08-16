#!/usr/bin/env python3
"""
overconfidence.py — A defined cut for "confidently wrong", and the evidence
that the finding does not live or die by that cut.

MURU-BENCH's overconfidence rate has always been "stated confidence above 0.7
on an answer that missed the ground-truth interval". That is a threshold, and
a threshold invites the obvious objection: pick a different one and the number
moves, so which number is the finding? This module answers that by reporting
the cut explicitly, sweeping it, and pairing it with two statistics that have
no cut at all.

Three things make the sweep non-decorative here:

  * **Verbalized confidence is a point mass, not a distribution.** Every model
    in the panel puts 20-70% of its answers on exactly 0.95, and the rest on
    0.5/0.8/0.9/1.0. The overconfidence rate is therefore a step function of
    the threshold, and a cut placed *on* a spike (0.9, 0.95) is not a
    measurement — it is a coin flip over which side of the tie convention you
    landed on. The sweep reports the tie mass sitting exactly at each cut so
    that a reader can see when this is happening.
  * **Strict vs inclusive matters at the spikes and nowhere else.** At the
    canonical 0.7 the two conventions differ by a handful of items; at 0.95
    they differ by hundreds. Reporting one convention without the other would
    hide the entire sensitivity in the place it is largest.
  * **Two different questions share the name "overconfidence".** The
    *population* rate ("what fraction of the whole split is a confident
    error") is the benchmark-level quantity; the *conditional* rate ("given a
    confident answer, how often is it wrong") is what a downstream user
    actually faces. They rank models differently when coverage differs, so
    both are reported.

The claim the paper rests on is comparative — overconfidence collapses from
the weakest model to the strongest — so what has to be threshold-robust is the
*ratio and the ordering*, not the absolute rate. ``panel_cut_dependence``
checks exactly that across the sweep.
"""

# The published cut. Strict inequality (conf > tau), which is the convention
# every number in the paper was computed under; ``inclusive=True`` gives the
# conf >= tau variant, reported alongside rather than silently swapped in.
CANONICAL_THRESHOLD = 0.7

# Spans "leaning confident" to "essentially certain". 0.9 and 0.95 sit on the
# panel's two largest confidence spikes and are included *because* they are the
# adversarial choices, not despite it.
THRESHOLD_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

# A cut with this much of some model's mass sitting exactly on it is decided by
# the tie convention rather than by the model.
TIE_DEGENERACY = 0.25

# Below this many confident errors for the *thinnest* model at a cut, the
# cross-model ordering at that cut is counting a handful of events.
MIN_EVENTS = 10


def _above(conf_value, tau, inclusive):
    return conf_value >= tau if inclusive else conf_value > tau


# ── Threshold-dependent statistics ─────────────────────────────────────


def overconfidence_rate(correct, conf, tau=CANONICAL_THRESHOLD, inclusive=False):
    """Share of *all* answered problems that are confident and wrong.

    This is the benchmark-level reading: how much of the split does the model
    get wrong while asserting it has it right. Denominator is every answered
    problem, so a model cannot improve it by declining to be confident on the
    items it would have failed — that shows up as a lower rate for a real
    reason.
    """
    if not correct:
        return 0.0
    hits = sum(
        1 for y, c in zip(correct, conf) if not y and _above(c, tau, inclusive)
    )
    return hits / len(correct)


def high_confidence_error_rate(correct, conf, tau=CANONICAL_THRESHOLD, inclusive=False):
    """Error rate *among* the answers the model stated confidently.

    The user-facing reading: if you act on the model's confident answers, this
    is how often you are acting on a wrong one. Returns ``(rate, n_confident)``
    — the n matters, because a model that is confident twice and wrong once
    scores 50% off two events.
    """
    idxs = [i for i, c in enumerate(conf) if _above(c, tau, inclusive)]
    if not idxs:
        return None, 0
    wrong = sum(1 for i in idxs if not correct[i])
    return wrong / len(idxs), len(idxs)


def confident_given_error(correct, conf, tau=CANONICAL_THRESHOLD, inclusive=False):
    """P(states high confidence | the answer is wrong) — the metacognitive term.

    The overconfidence rate factors exactly as

        rate = P(wrong) x P(confident | wrong)

    and only the second factor is about metacognition; the first is just
    accuracy. Reporting the factorisation is what distinguishes "this model is
    better at knowing when it is wrong" from "this model is wrong less often
    and the rate followed". Returns ``(rate, n_wrong)``.
    """
    wrong = [c for y, c in zip(correct, conf) if not y]
    if not wrong:
        return None, 0
    return sum(1 for c in wrong if _above(c, tau, inclusive)) / len(wrong), len(wrong)


def tie_mass(conf, tau):
    """Fraction of answers stating *exactly* the threshold confidence.

    The size of the block that the tie convention moves from one side of the
    cut to the other. Large values are the signal that a reported rate at this
    threshold is a convention artefact rather than a property of the model.
    """
    if not conf:
        return 0.0
    return sum(1 for c in conf if c == tau) / len(conf)


def threshold_sweep(correct, conf, grid=THRESHOLD_GRID):
    """Every threshold-dependent statistic at every cut, both conventions."""
    rows = []
    for tau in grid:
        strict_cond, n_strict = high_confidence_error_rate(correct, conf, tau, False)
        incl_cond, n_incl = high_confidence_error_rate(correct, conf, tau, True)
        rows.append({
            "tau": tau,
            "n": len(correct),
            "rate_strict": overconfidence_rate(correct, conf, tau, False),
            "rate_inclusive": overconfidence_rate(correct, conf, tau, True),
            "conditional_error_strict": strict_cond,
            "conditional_error_inclusive": incl_cond,
            "n_confident_strict": n_strict,
            "n_confident_inclusive": n_incl,
            "tie_mass": tie_mass(conf, tau),
        })
    return rows


# ── Threshold-free companions ──────────────────────────────────────────


def mean_overconfidence(correct, conf):
    """E[max(0, confidence - outcome)] — the cut-free version of the rate.

    With a binary outcome this is the mean stated confidence carried by the
    wrong answers, averaged over the whole split. Where the rate counts *how
    many* confident errors there are past an arbitrary line, this weighs each
    error by how confidently it was asserted, so no line is needed. It moves
    for the same reasons the rate does and can be quoted when a reviewer
    objects to the threshold at all.
    """
    if not correct:
        return 0.0
    return sum(max(0.0, c - (1 if y else 0)) for y, c in zip(correct, conf)) / len(correct)


def signed_calibration_gap(correct, conf):
    """Mean confidence minus accuracy: net over- (positive) or under-confidence.

    Coarser than ECE — it cancels rather than accumulating |gap| per bin — but
    it is the one number that says which *direction* a model errs in, which
    ECE by construction cannot.
    """
    if not correct:
        return 0.0
    n = len(correct)
    return sum(conf) / n - sum(correct) / n


def confidence_on_error(correct, conf):
    """Mean stated confidence on the answers that were wrong.

    Near-zero-information confidence shows up here as a value close to the
    model's overall mean confidence: it asserts its errors exactly as strongly
    as its successes.
    """
    wrong = [c for y, c in zip(correct, conf) if not y]
    if not wrong:
        return None
    return sum(wrong) / len(wrong)


def summary(correct, conf, grid=THRESHOLD_GRID):
    """Canonical rate, full sweep, and the cut-free companions for one model."""
    cond, n_conf = high_confidence_error_rate(correct, conf, CANONICAL_THRESHOLD)
    cge, n_wrong = confident_given_error(correct, conf, CANONICAL_THRESHOLD)
    return {
        "threshold": CANONICAL_THRESHOLD,
        "convention": "strict",
        "rate": overconfidence_rate(correct, conf, CANONICAL_THRESHOLD),
        "conditional_error_rate": cond,
        "n_confident": n_conf,
        "confident_given_error": cge,
        "n_wrong": n_wrong,
        "error_rate": 1.0 - (sum(correct) / len(correct)) if correct else 0.0,
        "mean_overconfidence": mean_overconfidence(correct, conf),
        "signed_calibration_gap": signed_calibration_gap(correct, conf),
        "mean_confidence_on_error": confidence_on_error(correct, conf),
        "sweep": threshold_sweep(correct, conf, grid),
    }


# ── Panel-level cut dependence ─────────────────────────────────────────


def spearman(xs, ys):
    """Spearman rank correlation with midranks for ties.

    scipy is deliberately not a dependency of this repo — every statistic in
    the paper is computed from stdlib so the archives reproduce on a bare
    interpreter — so the rank correlation lives here.
    """
    n = len(xs)
    if n < 2:
        return None

    def ranks(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if not dx or not dy:
        return None
    # Rounded and clamped: an exactly-agreeing ordering comes out of the sum of
    # products as 0.9999999999999998, and "almost exactly 1" is a
    # floating-point artefact rather than a finding — the stability verdict
    # below compares against 1.0 exactly.
    return max(-1.0, min(1.0, round(num / (dx * dy), 12)))


def panel_cut_dependence(rates_by_model, grid=THRESHOLD_GRID):
    """Is the cross-model overconfidence story a property of the cut?

    ``rates_by_model`` maps a model label to its ``threshold_sweep`` rows. For
    each threshold this returns the panel's spread (max/min rate and their
    ratio) and the rank correlation of the model ordering against the ordering
    at the canonical cut. A correlation of 1.0 everywhere means the sweep moves
    all models together and only the absolute level is threshold-dependent —
    which is the condition under which a comparative claim survives.

    Two kinds of cut are marked ``degenerate`` and excluded from that verdict,
    because at them the statistic has stopped being about confidence:

      * the cut lands on a confidence value that a large share of the panel's
        answers state *exactly*, so the tie convention alone moves hundreds of
        items across the line; and
      * the cut is so high that the models with the fewest confident errors
        have only a handful of them left, and their ordering is counting noise.

    The ratio is reported as ``None`` when the strongest model's rate is 0: a
    collapse to exactly zero is a stronger result than any finite multiple, and
    printing "inf" invites a reader to think the code divided by accident.
    """
    labels = sorted(rates_by_model)
    if not labels:
        return {"per_threshold": [], "orderings_agree": None}

    by_tau = {}
    for label in labels:
        for row in rates_by_model[label]:
            by_tau.setdefault(row["tau"], {})[label] = row

    canonical = by_tau.get(CANONICAL_THRESHOLD, {})
    base = [canonical[l]["rate_strict"] for l in labels] if canonical else []

    def corr(series):
        if not base or any(b is None for b in base):
            return None
        return spearman(base, series)

    per_threshold = []
    for tau in grid:
        vals = by_tau.get(tau)
        if not vals:
            continue
        strict = [vals[l]["rate_strict"] for l in labels]
        incl = [vals[l]["rate_inclusive"] for l in labels]
        lo, hi = min(strict), max(strict)
        per_threshold.append({
            "tau": tau,
            "min": lo,
            "max": hi,
            "ratio": (hi / lo) if lo > 0 else None,
            "spread_pp": 100 * (hi - lo),
            "rank_corr_vs_canonical": corr(strict),
            "rank_corr_inclusive_vs_canonical": corr(incl),
            "max_tie_mass": max(vals[l]["tie_mass"] for l in labels),
            "min_events": min(
                round(vals[l]["rate_strict"] * vals[l]["n"]) for l in labels
            ),
            "degenerate": (
                max(vals[l]["tie_mass"] for l in labels) >= TIE_DEGENERACY
                or min(round(vals[l]["rate_strict"] * vals[l]["n"]) for l in labels)
                < MIN_EVENTS
            ),
        })

    informative = [r for r in per_threshold if not r["degenerate"]]
    corrs = [
        r["rank_corr_vs_canonical"]
        for r in informative
        if r["rank_corr_vs_canonical"] is not None
    ]
    ratios = [r["ratio"] for r in per_threshold if r["ratio"] is not None]
    return {
        "models": labels,
        "per_threshold": per_threshold,
        "informative_range": [r["tau"] for r in informative],
        "degenerate_cuts": [r["tau"] for r in per_threshold if r["degenerate"]],
        "min_rank_corr_informative": min(corrs) if corrs else None,
        "orderings_agree": all(c == 1.0 for c in corrs),
        "min_ratio_all_cuts": min(ratios) if ratios else None,
    }
