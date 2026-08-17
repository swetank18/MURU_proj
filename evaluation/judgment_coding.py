#!/usr/bin/env python3
"""
judgment_coding.py — the half of the codebook a rule cannot decide.

``failure_codebook.py`` decides five codes from the record alone and refuses to
guess at the other seven, which is why 83% of the surviving error corpus comes
back uncoded. Those seven (R1, C1--C5, M1) require reading a reasoning chain
and deciding what the model was doing, and that is a judgment. This module is
the apparatus for making such judgments auditable rather than authoritative.

Three commitments, and they are the reason the file exists at all:

  * **One item, one primary code.** A wrong answer usually has several things
    wrong with it; the primary code is the *first* place the derivation left
    the rails, because that is the claim a taxonomy is actually making.
    Secondary codes are recorded but excluded from agreement statistics, since
    a multi-label chance-corrected coefficient measures something else.
  * **Every code carries a verbatim quote** from the response that licenses
    it. A coder who cannot quote the evidence has not coded the item, and a
    reader who disagrees can check the quote against the archive rather than
    taking the label on trust.
  * **Agreement is computed, not asserted.** ``cohens_kappa`` runs the moment a
    second coding file exists for the same items. Until one does, the reported
    result is one coder's opinion and this module says so in its own output.

Coding files live in ``evaluation/coding/`` and are committed, because unlike
every other artifact in this repository they cannot be regenerated: they are
the record of somebody's reading.

Usage:
  python evaluation/judgment_coding.py                      # report + matrix
  python evaluation/judgment_coding.py --against human      # add agreement
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.failure_codebook import (
    CODE_LABELS,
    ITEM_DEFECT_CODES,
    JUDGMENT_CODES,
    MECHANICAL_CODES,
)

CODING_DIR = PROJECT_ROOT / "evaluation" / "coding"
SAMPLE = PROJECT_ROOT / "evaluation" / "errors" / "sample_86.json"

# A coder may also assign a mechanical code as primary: the rules are narrow by
# design and miss cases a reader catches (an off-schema answer the parser
# recovered, a unit slip with no interval to corroborate it).
VALID_CODES = (
    tuple(JUDGMENT_CODES) + tuple(MECHANICAL_CODES)
    + tuple(ITEM_DEFECT_CODES) + ("XX",)
)

# XX: the response is present but says nothing that licenses any code --- the
# model asserts an answer with no derivation to inspect. Kept distinct from an
# uncoded item, which means the coder has not looked yet.
CODE_LABELS_EXTENDED = CODE_LABELS | {"XX": "unanalysable"}


def key(record):
    return f"{record['model_slug']}:{record['problem_id']}"


def load_sample(path=SAMPLE):
    with open(path) as f:
        return json.load(f)["records"]


def load_coding(name):
    """Load one coder's file by short name (e.g. 'claude' -> coding_claude.json)."""
    path = CODING_DIR / f"coding_{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def coded_map(coding):
    """{item key: primary code} for one coding file."""
    return {c["key"]: c["primary"] for c in coding["codes"]}


def validate(coding, sample):
    """Structural checks. Returns a list of complaints, empty if clean."""
    problems = []
    valid_keys = {key(r) for r in sample}
    seen = set()
    for c in coding["codes"]:
        k = c.get("key")
        if k not in valid_keys:
            problems.append(f"{k}: not in the sample")
        if k in seen:
            problems.append(f"{k}: coded twice")
        seen.add(k)
        if c.get("primary") not in VALID_CODES:
            problems.append(f"{k}: primary '{c.get('primary')}' is not a code")
        for s in c.get("secondary", []):
            if s not in VALID_CODES:
                problems.append(f"{k}: secondary '{s}' is not a code")
        if c.get("primary") != "XX" and not c.get("evidence", "").strip():
            problems.append(f"{k}: no evidence quote")
    missing = valid_keys - seen
    if missing:
        problems.append(f"{len(missing)} sample items uncoded")
    return problems


def cohens_kappa(a, b):
    """Cohen's kappa for two nominal codings of the same items.

    ``a`` and ``b`` are {key: code} maps; only keys present in both count.
    Returns (kappa, n, observed_agreement, expected_agreement). Kappa is
    undefined when both coders use exactly one code for everything and agree
    (expected agreement 1.0); that case returns None rather than dividing by
    zero, because perfect agreement on a constant is not evidence of anything.
    """
    shared = sorted(set(a) & set(b))
    n = len(shared)
    if n == 0:
        return None, 0, None, None
    agree = sum(1 for k in shared if a[k] == b[k])
    po = agree / n
    ca, cb = Counter(a[k] for k in shared), Counter(b[k] for k in shared)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return None, n, po, pe
    return (po - pe) / (1 - pe), n, po, pe


def confusion(a, b):
    """{(code_a, code_b): count} over shared items, for reading disagreements."""
    shared = set(a) & set(b)
    return Counter((a[k], b[k]) for k in sorted(shared))


def matrix(coding, sample):
    """{model: {code: n}} over primary codes."""
    by_key = {key(r): r for r in sample}
    out = defaultdict(Counter)
    for c in coding["codes"]:
        record = by_key.get(c["key"])
        if record:
            out[record["model"]][c["primary"]] += 1
    return out


def group_of(code):
    """Which half of the codebook a code belongs to, for the summary line."""
    if code in ITEM_DEFECT_CODES:
        return "item defect"
    if code in ("R1", "R2", "R3", "R4", "R5"):
        return "reporting"
    if code in ("A1", "XX"):
        return "no computation"
    if code.startswith("C"):
        return "computation"
    if code.startswith("M"):
        return "confidence"
    if code == "S1":
        return "format"
    return "other"


GROUP_ORDER = ("computation", "reporting", "no computation", "item defect", "format")
MODEL_ORDER = (
    "GPT-OSS-120B", "Qwen3-32B", "Llama-4-Scout-17B",
    "Llama-3.3-70B", "Llama-3.1-8B",
)


def wilson(k, n, z=1.96):
    """Wilson score interval; the sample is far too small for a normal one."""
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def group_matrix(coding, sample):
    """{model: Counter(group)} over primary codes, plus a per-model total."""
    by_key = {key(r): r for r in sample}
    out = defaultdict(Counter)
    for c in coding["codes"]:
        record = by_key.get(c["key"])
        if record:
            out[record["model"]][group_of(c["primary"])] += 1
            out[record["model"]]["_n"] += 1
    return out


def write_latex(coding, sample, path=None):
    """Model x failure-group table, with a Wilson interval on the reporting share."""
    mat = group_matrix(coding, sample)
    lines = []
    for model in [m for m in MODEL_ORDER if m in mat]:
        row = mat[model]
        n = row["_n"]
        cells = " & ".join(
            f"{row[g]} ({100 * row[g] / n:.0f}\\%)" if row[g] else "---"
            for g in GROUP_ORDER
        )
        lo, hi = wilson(row["reporting"], n)
        lines.append(
            f"{model} & {n} & {cells} & [{100 * lo:.0f}, {100 * hi:.0f}] \\\\"
        )
    path = path or PROJECT_ROOT / "paper" / "tables" / "judgment_codes.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\\newcommand{\\judgmentcoderows}{%\n" + "\n".join(lines) + "}\n"
    )
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coder", default="claude", help="primary coding file")
    ap.add_argument("--against", help="second coder to compute agreement with")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    sample = load_sample()
    coding = load_coding(args.coder)
    if coding is None:
        print(f"No coding file for '{args.coder}' in "
              f"{CODING_DIR.relative_to(PROJECT_ROOT)}.", file=sys.stderr)
        return 1

    complaints = validate(coding, sample)
    if complaints:
        print("Coding file problems:", file=sys.stderr)
        for c in complaints[:20]:
            print(f"  {c}", file=sys.stderr)
        return 1

    codes = coded_map(coding)
    n = len(codes)
    counts = Counter(codes.values())

    print(f"Judgment coding by '{coding['coder']}' over {n} sampled errors\n")
    print(f"{'code':6s} {'label':22s} {'group':14s} {'n':>4s}   share")
    print("-" * 60)
    for code, k in counts.most_common():
        bar = "#" * round(30 * k / n)
        print(f"{code:6s} {CODE_LABELS_EXTENDED.get(code, '?'):22s} "
              f"{group_of(code):14s} {k:4d}   {bar} {100 * k / n:.0f}%")

    groups = Counter(group_of(c) for c in codes.values())
    print("\nBy group: " + ", ".join(
        f"{g} {v} ({100 * v / n:.0f}%)" for g, v in groups.most_common()
    ))

    print("\nModel x primary code")
    mat = matrix(coding, sample)
    present = [c for c, _ in counts.most_common()]
    head = f"{'model':20s} {'n':>3s} " + " ".join(f"{c:>4s}" for c in present)
    print(head)
    print("-" * len(head))
    for model in sorted(mat, key=lambda m: -sum(mat[m].values())):
        row = mat[model]
        print(f"{model:20s} {sum(row.values()):3d} "
              + " ".join(f"{row.get(c, 0):4d}" for c in present))

    if args.against:
        other = load_coding(args.against)
        if other is None:
            print(f"\nNo coding file for '{args.against}' — agreement not "
                  f"computed.", file=sys.stderr)
            return 1
        k, n_shared, po, pe = cohens_kappa(codes, coded_map(other))
        print(f"\nAgreement with '{other['coder']}' over {n_shared} shared items")
        print(f"  observed  {po:.3f}")
        print(f"  expected  {pe:.3f}")
        print(f"  Cohen's kappa  {'undefined' if k is None else f'{k:.3f}'}")
        disagreements = [(x, y, c) for (x, y), c in confusion(codes, coded_map(other)).items() if x != y]
        if disagreements:
            print(f"\n  Disagreements ({sum(c for _, _, c in disagreements)}):")
            for x, y, c in sorted(disagreements, key=lambda t: -t[2]):
                print(f"    {args.coder} {x:3s} vs {args.against} {y:3s}   {c}")
    else:
        print(f"\nOne coder. Cohen's kappa is not computed and no reliability "
              f"claim is\navailable from this file alone; pass --against "
              f"<coder> once a second\ncoding of the same items exists.")

    if not args.no_latex:
        print(f"Saved: {write_latex(coding, sample).relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
