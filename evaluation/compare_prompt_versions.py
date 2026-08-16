#!/usr/bin/env python3
"""
compare_prompt_versions.py — Does a prompt that states the unit convention
produce what the post-hoc correction predicts?

The published panel was collected under prompt version 1, which asked for "a
single number" and never said which unit. ``unit_accounting`` corrects that
after the fact by reading a prediction in the unit the model's own interval
implies. It is a defensible rule, but it is still a rule applied by us to data
we already had, and the paper now rests on it. The independent check is to ask
a model the same questions under a prompt that removes the ambiguity, and see
which of the two scorings the fresh answers agree with.

Three outcomes, and they mean different things:

  * v2 accuracy matches the **unit-aware** v1 score --- the correction
    recovered what an unambiguous prompt would have got, and the panel numbers
    stand as reported.
  * v2 accuracy matches the **raw** v1 score --- the models were wrong for
    reasons the correction excused, and the correction is too lenient.
  * v2 accuracy matches **neither** --- stating the unit changed behaviour
    beyond units (it is a longer prompt, and instruction-following is not
    free), which is a prompt-sensitivity finding in its own right and belongs
    in the robustness section rather than being read as validation.

Everything is paired: both arms answer the same 100 problems, so the test is
McNemar on the discordant pairs rather than a comparison of two independent
proportions. The comparison also reports how many unit mismatches remain under
v2, which is the direct measure of whether the prompt fix worked at all.

Usage:
  python evaluation/compare_prompt_versions.py
  python evaluation/compare_prompt_versions.py --tag promptv2
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.aggregate_real_llm import (
    DISPLAY_NAMES,
    find_latest_result,
    load_problems,
)
from evaluation.bootstrap_analysis import mcnemar_test
from evaluation import unit_accounting as ua

ABLATIONS = PROJECT_ROOT / "evaluation" / "baselines" / "ablations"
SUBSET = PROJECT_ROOT / "data" / "robustness_subset.json"


def load_arm(slug, tag):
    """Latest tagged ablation archive for a model, if any."""
    matches = sorted(ABLATIONS.glob(f"{slug}_*_{tag}.json"))
    if not matches:
        return None
    with open(matches[-1]) as f:
        return json.load(f)


def predictions_by_id(data):
    """{problem_id: (point_estimate, interval, confidence)} for readable answers."""
    out = {}
    for r in data.get("raw_results", []):
        if not r.get("success"):
            continue
        parsed = r.get("parsed") or {}
        if parsed.get("point_estimate") is None:
            continue
        interval = parsed.get("confidence_interval")
        out[r["problem_id"]] = (
            parsed["point_estimate"],
            tuple(interval) if interval else None,
            parsed.get("confidence", 0.5),
        )
    return out


def score(preds, problems_by_id, ids):
    """Raw and unit-aware correctness vectors over ``ids``, plus mismatch count."""
    raw, unit, mismatches = [], [], 0
    for pid in ids:
        pe, interval, _ = preds[pid]
        problem = problems_by_id[pid]
        status = ua.classify_prediction(pe, interval, problem)
        raw.append(int(status == "correct"))
        unit.append(int(ua.is_correct(status, unit_aware=True)))
        mismatches += int(status == "unit_mismatch")
    return raw, unit, mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="promptv2")
    args = ap.parse_args()

    if not SUBSET.exists():
        print("Run scripts/robustness_subset.py first.", file=sys.stderr)
        return 1
    with open(SUBSET) as f:
        subset_ids = json.load(f)["problem_ids"]

    problems = load_problems(PROJECT_ROOT / "data" / "test")
    problems_by_id = {p["id"]: p for p in problems}

    rows = []
    for slug, display in DISPLAY_NAMES.items():
        arm = load_arm(slug, args.tag)
        if not arm:
            continue
        panel_path = find_latest_result(slug)
        if not panel_path:
            continue
        with open(panel_path) as f:
            panel = json.load(f)

        v2 = predictions_by_id(arm)
        v1 = predictions_by_id(panel)
        # Pair strictly: only problems both arms answered readably.
        ids = [p for p in subset_ids if p in v1 and p in v2]
        if not ids:
            continue

        v1_raw, v1_unit, v1_mm = score(v1, problems_by_id, ids)
        v2_raw, v2_unit, v2_mm = score(v2, problems_by_id, ids)

        rows.append({
            "display": display,
            "n": len(ids),
            "v1_raw": sum(v1_raw) / len(ids),
            "v1_unit": sum(v1_unit) / len(ids),
            "v2_raw": sum(v2_raw) / len(ids),
            "v2_unit": sum(v2_unit) / len(ids),
            "v1_mismatches": v1_mm,
            "v2_mismatches": v2_mm,
            # The decisive test: v2 as-scored against v1 as-corrected. A null
            # result here is the good outcome -- it says the correction landed
            # where an unambiguous prompt lands.
            # mcnemar_test returns (b_only, a_only, two_sided_p): here
            # b_only counts problems v2 got right that v1-corrected did not.
            "mcnemar_vs_v1_unit": dict(zip(
                ("v2_only", "v1_only", "p_value"), mcnemar_test(v1_unit, v2_raw)
            )),
            "mcnemar_vs_v1_raw": dict(zip(
                ("v2_only", "v1_only", "p_value"), mcnemar_test(v1_raw, v2_raw)
            )),
            "n_v2_answered": len(v2),
            "n_subset": len(subset_ids),
        })

    if not rows:
        print(f"No ablation arms tagged '{args.tag}' found in "
              f"{ABLATIONS.relative_to(PROJECT_ROOT)}.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: -r["v1_unit"])
    print(f"Prompt v1 (archived panel) vs v2 (unit convention stated), "
          f"paired on the robustness subset\n")
    header = (f"{'model':20s} {'n':>4s} {'v1 raw':>8s} {'v1 unit':>8s} "
              f"{'v2 raw':>8s} {'v2 unit':>8s} {'v1 UM':>6s} {'v2 UM':>6s} "
              f"{'McNemar p (v2 vs v1-unit)':>26s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        mc = r["mcnemar_vs_v1_unit"]
        verdict = f"{mc['p_value']:.3f}  (+{mc['v2_only']}/-{mc['v1_only']})"
        print(f"{r['display']:20s} {r['n']:4d} "
              f"{100*r['v1_raw']:7.1f}% {100*r['v1_unit']:7.1f}% "
              f"{100*r['v2_raw']:7.1f}% {100*r['v2_unit']:7.1f}% "
              f"{r['v1_mismatches']:6d} {r['v2_mismatches']:6d} "
              f"{verdict:>26s}")

    print("\nReading:")
    print("  v2 raw ≈ v1 unit   → the post-hoc correction recovers what an "
          "unambiguous prompt gets.")
    print("  v2 raw ≈ v1 raw    → the correction is too lenient.")
    print("  v2 UM near zero    → stating the convention fixed the root cause.")

    out = ABLATIONS / f"compare_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
