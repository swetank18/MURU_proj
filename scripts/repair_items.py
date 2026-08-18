#!/usr/bin/env python3
"""
repair_items.py — build the v1.1 errata set from the v1.0 corpus.

`data/` is v1.0: the corpus the model panel actually answered. It is not
repaired in place. The committed archives hold responses to those stems, four of
the five panel endpoints have since been withdrawn, and rewriting a stem would
leave an archived answer attached to a question nobody asked — so v1.0 is tagged
as answered and stays reproducible, and the repairs ship alongside as v1.1.

This script regenerates every problem `scripts/audit_item_defects.py` flags,
using the fixed generator, and writes the replacements to `errata/v1.1/`. Each
replacement keeps the original's id, difficulty and split, is drawn from a seed
derived from its own id (so the set is reproducible), and is checked to be
schema-valid, audit-clean, and in the same category as the item it replaces.

v1.1 is defined as: v1.0, with each file in `errata/v1.1/` replacing the file of
the same id. `--apply` performs that substitution against `data/` for anyone who
wants the materialised tree; the repository keeps `data/` at v1.0.

Usage:
    python scripts/repair_items.py              # write errata/v1.1/
    python scripts/repair_items.py --dry-run    # report, write nothing
    python scripts/repair_items.py --apply      # materialise v1.1 into data/

Exit codes:
    0 — every flagged item was repaired
    1 — at least one could not be
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema is required. Install with: pip install jsonschema>=4.20.0")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_item_defects as audit
import generate_problems as gp

ERRATA_VERSION = "v1.1"
BASE_VERSION = "v1.0"
ERRATA_DIR = PROJECT_ROOT / "errata" / ERRATA_VERSION
SCHEMA_PATH = PROJECT_ROOT / "problem_schema.json"

# A regenerated item is drawn from a seed derived from its own id, so the set is
# reproducible and independent of the order the items are processed in. The salt
# ties the draw to this errata version: a future v1.2 regenerating the same id
# gets a different item rather than silently the same one.
SEED_SALT = f"muru-errata-{ERRATA_VERSION}"

# Redraws allowed per item before giving up. A clean draw is the overwhelmingly
# common case; the budget exists so that a template whose constraints cannot be
# met reports that rather than looping.
MAX_ATTEMPTS = 64


def template_by_author() -> dict:
    """Map metadata.author to the template that produces it.

    Read off the generator rather than hard-coded, because the two names do not
    always agree — the `decision_payoff` template signs its items
    `generator_decision`.
    """
    mapping = {}
    state = random.getstate()
    random.seed(0)
    try:
        for name, template in gp.TEMPLATES.items():
            probe = template.generate(1, template.difficulty_range[0])
            mapping[probe["metadata"]["author"]] = name
    finally:
        random.setstate(state)
    return mapping


def load_corpus() -> list:
    problems = []
    for split in audit.SPLITS:
        directory = PROJECT_ROOT / "data" / split
        for path in sorted(directory.glob("MURU-*.json")):
            with open(path) as handle:
                problem = json.load(handle)
            problem["_split"] = split
            problem["_path"] = path
            problems.append(problem)
    return problems


def defects_of(problem: dict) -> list:
    return [message for check in audit.CHECKS for message in check(problem)]


def regenerate(original: dict, template_name: str, schema: dict):
    """Draw a defect-free replacement for `original`. Returns (item, attempts)."""
    template = gp.TEMPLATES[template_name]
    numeric_id = int(original["id"].split("-")[1])
    state = random.getstate()
    try:
        for attempt in range(MAX_ATTEMPTS):
            random.seed(f"{SEED_SALT}:{original['id']}:{attempt}")
            try:
                candidate = template.generate(numeric_id, original["difficulty"])
            except Exception:
                continue

            # The replacement has to be interchangeable with what it replaces:
            # same id, same difficulty, same category, or it silently changes
            # the composition of a split that was stratified on all three.
            if candidate["id"] != original["id"]:
                continue
            if candidate["difficulty"] != original["difficulty"]:
                continue
            if candidate["category"] != original["category"]:
                continue

            probe = dict(candidate)
            probe["_split"] = original["_split"]
            if defects_of(probe):
                continue
            try:
                jsonschema.validate(candidate, schema)
            except jsonschema.ValidationError:
                continue
            return candidate, attempt + 1
    finally:
        random.setstate(state)
    return None, MAX_ATTEMPTS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Also copy the replacements over data/, materialising v1.1 in place.",
    )
    args = parser.parse_args()

    with open(SCHEMA_PATH) as handle:
        schema = json.load(handle)

    by_author = template_by_author()
    corpus = load_corpus()
    flagged = [(p, defects_of(p)) for p in corpus]
    flagged = [(p, d) for p, d in flagged if d]

    print(f"\n  {len(corpus)} problems in {BASE_VERSION}; {len(flagged)} carry a defect\n")
    if not flagged:
        print("  Nothing to repair.\n")
        return 0

    repaired, failures = [], []
    for original, messages in flagged:
        author = original["metadata"].get("author", "")
        template_name = by_author.get(author)
        if template_name is None:
            failures.append((original, messages, f"no template produces author '{author}'"))
            continue
        candidate, attempts = regenerate(original, template_name, schema)
        if candidate is None:
            failures.append((original, messages, f"no clean draw in {MAX_ATTEMPTS} attempts"))
            continue
        repaired.append(
            {
                "id": original["id"],
                "split": original["_split"],
                "template": template_name,
                "defects": [m.split(":", 1)[0] for m in messages],
                "messages": messages,
                "attempts": attempts,
                "item": candidate,
                "original_path": original["_path"],
            }
        )

    by_split = Counter(r["split"] for r in repaired)
    by_class = Counter(c for r in repaired for c in r["defects"])
    by_template = Counter(r["template"] for r in repaired)
    print(f"  repaired {len(repaired)} / {len(flagged)}")
    print("  by split:    " + ", ".join(f"{k}={v}" for k, v in sorted(by_split.items())))
    print("  by class:    " + ", ".join(f"{k}={v}" for k, v in sorted(by_class.items())))
    print("  by template: " + ", ".join(f"{k}={v}" for k, v in sorted(by_template.items())))
    if failures:
        print(f"\n  {len(failures)} could NOT be repaired:")
        for original, _messages, reason in failures:
            print(f"    {original['id']}  {reason}")

    if args.dry_run:
        print("\n  --dry-run: nothing written.\n")
        return 1 if failures else 0

    if ERRATA_DIR.exists():
        shutil.rmtree(ERRATA_DIR)
    ERRATA_DIR.mkdir(parents=True)

    entries = []
    for record in sorted(repaired, key=lambda r: r["id"]):
        item = dict(record["item"])
        item["metadata"] = dict(item["metadata"])
        item["metadata"]["errata"] = {
            "version": ERRATA_VERSION,
            "replaces": BASE_VERSION,
            "defects": sorted(set(record["defects"])),
        }
        out = ERRATA_DIR / f"{record['id']}.json"
        out.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n")
        entries.append(
            {
                "id": record["id"],
                "split": record["split"],
                "template": record["template"],
                "defects": sorted(set(record["defects"])),
                "messages": record["messages"],
                "sha256_v1_0": sha256(record["original_path"]),
                "sha256_v1_1": sha256(out),
            }
        )

    manifest = {
        "version": ERRATA_VERSION,
        "replaces": BASE_VERSION,
        "description": (
            f"{ERRATA_VERSION} is {BASE_VERSION} with each file listed here replacing the "
            f"problem of the same id. {BASE_VERSION} is the corpus the model panel answered "
            f"and is left intact in data/ so the published numbers stay reproducible."
        ),
        "audit": "scripts/audit_item_defects.py",
        "seed_salt": SEED_SALT,
        "n_base": len(corpus),
        "n_replaced": len(entries),
        "by_split": dict(sorted(by_split.items())),
        "by_defect_class": dict(sorted(by_class.items())),
        "by_template": dict(sorted(by_template.items())),
        "items": entries,
    }
    (ERRATA_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  Wrote {len(entries)} replacements + MANIFEST.json to "
          f"{ERRATA_DIR.relative_to(PROJECT_ROOT)}/")

    if args.apply:
        for record in repaired:
            src = ERRATA_DIR / f"{record['id']}.json"
            shutil.copyfile(src, record["original_path"])
        print(f"  --apply: copied {len(repaired)} replacements over data/ "
              f"(data/ is now {ERRATA_VERSION})")
    print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
