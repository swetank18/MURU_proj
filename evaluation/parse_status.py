#!/usr/bin/env python3
"""
parse_status.py — Separate parse failure from wrong answer.

Every row in a MURU-BENCH archive is one of three different things, and the
leaderboard is only meaningful if they are kept apart:

  1. the model answered and the answer was scored (``ok``);
  2. the model answered but the harness could not read the answer
     (``truncated`` / ``missing_field`` / ``no_schema`` / ``refused``) —
     a *model* failure, and one that must not be silently dropped, because
     dropping it turns a format failure into a free pass;
  3. the request never produced a response (``endpoint_unavailable`` /
     ``rate_limited`` / ``timeout`` / ``api_error`` / ``unattempted``) —
     a *provider* failure, which is missing data and carries no information
     about the model.

Class 2 is scored as incorrect under the strict accounting; class 3 is
excluded from both accountings. The distinction matters: under the previous
"exclude everything that did not parse" rule, Llama-3.1-8B's 12 unreadable
responses were removed from its denominator, which flattered a model whose
failure mode is precisely that it runs out of tokens mid-derivation.

Statuses are derived from the committed archives, so this is a re-analysis
of existing runs — no re-querying required.

Usage:
    python evaluation/parse_status.py            # report + write JSON
    python evaluation/parse_status.py --quiet    # write JSON only
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Status vocabulary ──────────────────────────────────────────────────
# Model-side statuses count against the model under strict accounting.
MODEL_FAILURE_STATUSES = (
    "truncated",
    "missing_field",
    "format_variant",
    "no_schema",
    "refused",
)
# Provider-side statuses are missing data under both accountings.
PROVIDER_FAILURE_STATUSES = (
    "endpoint_unavailable",
    "rate_limited",
    "timeout",
    "api_error",
    "empty_response",
    "unattempted",
)
ALL_STATUSES = ("ok",) + MODEL_FAILURE_STATUSES + PROVIDER_FAILURE_STATUSES

# A response that stops mid-thought does not end on terminal punctuation.
# Truncation at max_tokens is the dominant cause on small models.
_TERMINAL_CHARS = ".!?\"')]}>"

_REFUSAL_PATTERNS = re.compile(
    r"\b(i cannot|i can't|i am unable to|i'm unable to|as an ai|i do not have enough|"
    r"i won't be able to|cannot provide an answer|refuse to)\b",
    re.IGNORECASE,
)

# The canonical marker the prompt asks for, and the looser variants models
# reach for instead (markdown bolding, spaced words, colon-less headers).
_CANONICAL_MARKER = re.compile(r"POINT_ESTIMATE\s*:", re.IGNORECASE)
_VARIANT_WITH_VALUE = re.compile(
    r"point[\s_*]{0,4}estimate[\s*:]{0,6}\s*([-+]?\d*\.?\d+)", re.IGNORECASE
)


def _looks_truncated(response: str) -> bool:
    """True if the generation appears to have been cut off mid-thought.

    Terminal punctuation is the primary signal. The trailing-decimal case
    ("... 1.96 * 0.") has to be special-cased: it ends in a period but is a
    number sliced in half, not a finished sentence.
    """
    tail = response.rstrip()
    if not tail:
        return True
    if tail[-1] not in _TERMINAL_CHARS:
        return True
    return tail[-1] == "." and len(tail) > 1 and tail[-2].isdigit()


def classify_entry(entry: dict) -> str:
    """Assign one status from ALL_STATUSES to a single archive record."""
    parsed = entry.get("parsed") or {}
    if parsed.get("point_estimate") is not None:
        return "ok"

    err = (entry.get("error") or "").lower()
    if err:
        if "does not exist" in err or "model_not_found" in err or "404" in err:
            return "endpoint_unavailable"
        if "429" in err or "rate limit" in err or "rate_limit" in err or "tokens per day" in err:
            return "rate_limited"
        if "timeout" in err or "timed out" in err:
            return "timeout"
        return "api_error"

    response = entry.get("response") or ""
    if not response.strip():
        return "empty_response"

    if _REFUSAL_PATTERNS.search(response[-1200:]):
        return "refused"

    # The canonical schema block was emitted but the point-estimate slot was
    # empty or unreadable: the model followed the format and still gave us no
    # answer.
    if _CANONICAL_MARKER.search(response):
        return "missing_field"

    # A readable number under a non-canonical label ("**Point Estimate:** 0.65").
    # The model did the work; it just did not honour the output contract, so
    # this is instruction-following failure rather than a missing answer.
    if _VARIANT_WITH_VALUE.search(response):
        return "format_variant"

    # No schema block at all. Ending mid-thought means the generation was cut
    # off; ending cleanly means the model simply ignored the output contract.
    if _looks_truncated(response):
        return "truncated"
    return "no_schema"


def status_report(archive: dict, n_test: int, test_ids: set[str] | None = None) -> dict:
    """Per-status counts and rates for one archive.

    ``parse_rate`` is over *attempted* problems (those the provider actually
    answered), which is the quantity that describes the model. ``coverage``
    is over the full test split and describes the run.
    """
    raw = archive.get("raw_results", [])
    per_problem = {}
    for entry in raw:
        pid = entry.get("problem_id")
        if test_ids is not None and pid not in test_ids:
            continue
        per_problem[pid] = classify_entry(entry)

    if test_ids is not None:
        for pid in test_ids - set(per_problem):
            per_problem[pid] = "unattempted"

    counts = Counter(per_problem.values())
    n_ok = counts["ok"]
    n_model_fail = sum(counts[s] for s in MODEL_FAILURE_STATUSES)
    n_provider_fail = sum(counts[s] for s in PROVIDER_FAILURE_STATUSES)
    n_attempted = n_ok + n_model_fail

    return {
        "n_test": n_test,
        "n_attempted": n_attempted,
        "n_parsed": n_ok,
        "n_model_failure": n_model_fail,
        "n_provider_failure": n_provider_fail,
        "parse_rate": (n_ok / n_attempted) if n_attempted else None,
        "coverage": n_attempted / n_test if n_test else None,
        "counts": {s: counts[s] for s in ALL_STATUSES if counts[s]},
        "per_problem": per_problem,
    }


# ── CLI ────────────────────────────────────────────────────────────────

def _load_test_ids() -> set[str]:
    return {p.stem for p in (PROJECT_ROOT / "data" / "test").rglob("MURU-*.json")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="suppress the stdout table")
    ap.add_argument(
        "--out",
        default="evaluation/baselines/parse_status.json",
        help="where to write the machine-readable report",
    )
    args = ap.parse_args()

    from evaluation.aggregate_real_llm import DISPLAY_NAMES, find_latest_result

    test_ids = _load_test_ids()
    report = {}
    for slug, display in DISPLAY_NAMES.items():
        path = find_latest_result(slug)
        if not path:
            continue
        with open(path) as f:
            archive = json.load(f)
        if not archive.get("raw_results"):
            continue
        entry = status_report(archive, n_test=len(test_ids), test_ids=test_ids)
        entry["display"] = display
        entry["result_file"] = str(path.relative_to(PROJECT_ROOT))
        report[slug] = entry

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    if not args.quiet:
        print(f"\n{'Model':<20} {'Parsed':>7} {'Attempt':>8} {'Parse%':>8} {'Cover%':>8}  breakdown")
        print("─" * 96)
        for slug, r in report.items():
            breakdown = ", ".join(
                f"{s}={n}" for s, n in r["counts"].items() if s != "ok"
            ) or "—"
            print(
                f"{r['display']:<20} {r['n_parsed']:>7} {r['n_attempted']:>8} "
                f"{100 * r['parse_rate']:>7.1f}% {100 * r['coverage']:>7.1f}%  {breakdown}"
            )
        print()
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
