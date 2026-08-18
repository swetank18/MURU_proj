#!/usr/bin/env python3
"""
defect_leaveout.py — does the panel result survive dropping the broken items?

`scripts/audit_item_defects.py` finds 29 of the 301 test problems carrying an
item-construction defect: a physically impossible stem value, a base-rate stem
that states one accuracy and quotes another, or a ground-truth interval narrower
than the precision it invites. The items are not repaired in place — the
committed archives hold answers to the stems as they stand, and four of the five
panel endpoints have been withdrawn, so an edited stem would leave an archived
answer attached to a question nobody asked. v1.0 is therefore tagged as
answered, and the repairs ship as v1.1.

That leaves the question a reviewer will ask immediately: how much of the
published result rests on the 29. This script answers it by re-scoring every
archive on the clean 272 and reporting the difference. It is pure re-analysis of
the committed archives — no API key, no network — and it uses the same unit
accounting, the same bootstrap and the same ECE estimator as the main tables, so
the "all items" column here reproduces the leaderboard.

Usage:
  python evaluation/defect_leaveout.py            # report + write the table
  python evaluation/defect_leaveout.py --json     # machine-readable only
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_item_defects as audit
from evaluation.aggregate_real_llm import (
    DISPLAY_NAMES,
    N_BOOTSTRAP,
    find_latest_result,
    load_problems,
    parse_predictions_from_raw,
    unit_normalized,
)
from evaluation.bootstrap_analysis import ece_from_paired, per_problem_outcomes
from evaluation import overconfidence as oc

# Enough that a 29-item subset still gets a usable interval; the defective
# subset is small and its CI is wide on purpose.
SEED = 0xDEFEC7


def defective_ids(problems) -> set:
    """Problem ids in this set carrying at least one item-construction defect."""
    flagged = set()
    for problem in problems:
        for check in audit.CHECKS:
            if check(problem):
                flagged.add(problem["id"])
                break
    return flagged


def score(problems, preds):
    """Unit-aware accuracy / ECE / overconfidence with percentile bootstrap CIs."""
    if not problems:
        return None
    correct, _fw, overconf, conf = per_problem_outcomes(problems, preds)
    n = len(correct)
    rng = random.Random(SEED)
    acc_b, ece_b, oc_b = [], [], []
    for _ in range(N_BOOTSTRAP):
        idxs = [rng.randrange(n) for _ in range(n)]
        c = [correct[i] for i in idxs]
        p = [conf[i] for i in idxs]
        acc_b.append(sum(c) / n)
        ece_b.append(ece_from_paired(c, p))
        oc_b.append(sum(overconf[i] for i in idxs) / n)

    def ci(samples):
        samples.sort()
        return [samples[int(0.025 * N_BOOTSTRAP)], samples[int(0.975 * N_BOOTSTRAP)]]

    return {
        "n": n,
        "accuracy": {"point": sum(correct) / n, "ci95": ci(acc_b)},
        "ece": {"point": ece_from_paired(correct, conf), "ci95": ci(ece_b)},
        "overconfidence": {"point": sum(overconf) / n, "ci95": ci(oc_b)},
    }


def analyse():
    problems = load_problems(PROJECT_ROOT / "data" / "test")
    problems_by_id = {p["id"]: p for p in problems}
    flagged = defective_ids(problems)

    rows = []
    for slug, display in DISPLAY_NAMES.items():
        path = find_latest_result(slug)
        if not path:
            continue
        with open(path) as handle:
            data = json.load(handle)
        raw = data.get("raw_results", [])
        if not raw:
            continue
        preds = parse_predictions_from_raw(raw, problems_by_id)
        if not preds:
            continue
        preds, _ = unit_normalized(preds, problems_by_id)

        answered = {p.problem_id for p in preds}
        full = [p for p in problems if p["id"] in answered]
        clean = [p for p in full if p["id"] not in flagged]
        broken = [p for p in full if p["id"] in flagged]

        row = {
            "model": display,
            "archive": path.name,
            "all": score(full, preds),
            "clean": score(clean, preds),
            "defective": score(broken, preds),
        }
        row["delta_accuracy_pp"] = 100 * (
            row["clean"]["accuracy"]["point"] - row["all"]["accuracy"]["point"]
        )
        row["delta_ece"] = row["clean"]["ece"]["point"] - row["all"]["ece"]["point"]
        rows.append(row)

    rows.sort(key=lambda r: -r["all"]["accuracy"]["point"])

    # The headline is a null: accuracy and calibration-in-level are separable.
    # If dropping the defective items moved that, the null would be an artefact
    # of the broken items rather than a property of the panel.
    rho = {
        which: oc.spearman(
            [r[which]["accuracy"]["point"] for r in rows],
            [r[which]["ece"]["point"] for r in rows],
        )
        for which in ("all", "clean")
    }
    ranking = {
        which: [r["model"] for r in sorted(rows, key=lambda r: -r[which]["accuracy"]["point"])]
        for which in ("all", "clean")
    }

    return {
        "n_test": len(problems),
        "n_defective": len(flagged),
        "defective_ids": sorted(flagged),
        "rows": rows,
        "accuracy_ece_spearman": rho,
        "ranking": ranking,
        "ranking_preserved": ranking["all"] == ranking["clean"],
    }


def write_table(result) -> Path:
    lines = []
    for row in result["rows"]:
        a, c, d = row["all"], row["clean"], row["defective"]
        defective_cell = (
            f"{100 * d['accuracy']['point']:.1f}\\%" if d else "--"
        )
        lines.append(
            f"{row['model']} & {a['n']} & {100 * a['accuracy']['point']:.1f}\\% & "
            f"{c['n']} & {100 * c['accuracy']['point']:.1f}\\%~[{100 * c['accuracy']['ci95'][0]:.1f}, "
            f"{100 * c['accuracy']['ci95'][1]:.1f}] & {row['delta_accuracy_pp']:+.1f} & "
            f"{d['n'] if d else 0} & {defective_cell} & "
            f"{a['ece']['point']:.3f} & {c['ece']['point']:.3f} \\\\"
        )
    path = PROJECT_ROOT / "paper" / "tables" / "defect_leaveout.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\\newcommand{\\defectleaveoutrows}{%\n" + "\n".join(lines) + "}\n"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    result = analyse()

    out = PROJECT_ROOT / "evaluation" / "baselines" / "defect_leaveout.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        print()
        return 0

    n_clean = result["n_test"] - result["n_defective"]
    print(
        f"\n  Leave-out: {result['n_defective']} of {result['n_test']} test items carry an "
        f"item-construction defect; re-scoring on the clean {n_clean}.\n"
    )
    header = f"  {'model':22s} {'all':>16s} {'clean':>16s} {'delta':>8s} {'defective':>16s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in result["rows"]:
        a, c, d = row["all"], row["clean"], row["defective"]
        dcell = f"{100 * d['accuracy']['point']:5.1f}% (n={d['n']:2d})" if d else "        --"
        print(
            f"  {row['model']:22s} "
            f"{100 * a['accuracy']['point']:6.1f}% (n={a['n']:3d}) "
            f"{100 * c['accuracy']['point']:6.1f}% (n={c['n']:3d}) "
            f"{row['delta_accuracy_pp']:+7.1f} "
            f"{dcell:>16s}"
        )
    print()
    for row in result["rows"]:
        print(
            f"  {row['model']:22s} ECE {row['all']['ece']['point']:.3f} -> "
            f"{row['clean']['ece']['point']:.3f}  ({row['delta_ece']:+.3f})"
        )

    rho = result["accuracy_ece_spearman"]
    print(
        f"\n  accuracy/ECE Spearman: all items {rho['all']:+.2f}, "
        f"clean items {rho['clean']:+.2f}"
    )
    print(
        "  leaderboard ordering: "
        + ("unchanged" if result["ranking_preserved"] else "CHANGES on the clean subset")
    )
    print(f"\n  Wrote {out.relative_to(PROJECT_ROOT)}")
    print(f"  Wrote {write_table(result).relative_to(PROJECT_ROOT)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
