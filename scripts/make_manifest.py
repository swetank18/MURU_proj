#!/usr/bin/env python3
"""
make_manifest.py — Inventory of the committed evaluation archives.

"Every number reconstructs from the archives" is only a checkable claim if a
reader can tell which archives exist, which one backs each leaderboard row,
and whether the copy they have is the copy we measured. This writes a manifest
with a SHA-256 per archive, the parse-status breakdown, and a flag marking the
file each published row was computed from.

Usage:
    python scripts/make_manifest.py
"""

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.parse_status import status_report

BASELINES = PROJECT_ROOT / "evaluation" / "baselines"
SUMMARY = BASELINES / "real_llm_summary.json"
MANIFEST = BASELINES / "MANIFEST.json"

# Simulator-tier archives are deterministically regenerable from a seed and are
# not tracked, so they are not part of the evidentiary manifest.
SIMULATED_PREFIXES = (
    "random_baseline",
    "heuristic_baseline",
    "competent_model",
    "strong_model",
    "expert_model",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    test_ids = {p.stem for p in (PROJECT_ROOT / "data" / "test").rglob("MURU-*.json")}
    cited = set()
    if SUMMARY.exists():
        with open(SUMMARY) as f:
            # Underscore-prefixed keys hold panel-level analyses, not models.
            cited = {
                v["result_file"]
                for k, v in json.load(f).items()
                if not k.startswith("_")
            }

    entries = []
    for path in sorted(BASELINES.glob("*.json")):
        if path.name in ("MANIFEST.json", "real_llm_summary.json", "parse_status.json"):
            continue
        if path.name.startswith(SIMULATED_PREFIXES):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if not data.get("raw_results"):
            continue

        rel = str(path.relative_to(PROJECT_ROOT))
        status = status_report(data, n_test=len(test_ids), test_ids=test_ids)
        entries.append({
            "file": rel,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "model": data.get("model"),
            "timestamp": data.get("timestamp"),
            "seed": data.get("seed"),
            "resumed": bool(data.get("resumed")),
            "salvaged": bool(data.get("salvaged")) or "salvaged" in path.name,
            "n_records": len(data["raw_results"]),
            "n_parsed": status["n_parsed"],
            "parse_rate": status["parse_rate"],
            "status_counts": status["counts"],
            "cited_in_paper": rel in cited,
        })

    manifest = {
        "description": (
            "Inventory of committed MURU-BENCH evaluation archives. Archives "
            "flagged cited_in_paper back a published leaderboard row; the rest "
            "are earlier accumulation checkpoints of the same runs, retained so "
            "the coverage history is auditable."
        ),
        "schema": "results/schema.json",
        "n_archives": len(entries),
        "n_cited": sum(e["cited_in_paper"] for e in entries),
        "archives": entries,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"{len(entries)} archives, {manifest['n_cited']} cited in the paper")
    for e in entries:
        if e["cited_in_paper"]:
            print(
                f"  ★ {e['file']:<62} {e['n_parsed']:>3}/{e['n_records']:<3} "
                f"parsed  {e['sha256'][:12]}"
            )
    print(f"Saved: {MANIFEST.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
