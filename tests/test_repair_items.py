"""Tests for scripts/repair_items.py and the v1.1 errata set it builds.

v1.0 stays in data/ because the archives hold answers to those stems and four of
the five endpoints are gone. That makes the errata set the only place the fixes
live, so what has to be guaranteed here is that a replacement is genuinely
interchangeable with what it replaces — same id, same difficulty, same category,
same split — and that regenerating it twice gives the same item. A replacement
that quietly changed a problem's category would restratify a split that was
built on category and difficulty, and nobody would notice.
"""

import json
import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_item_defects as audit
import generate_problems as gp
import repair_items as repair

ERRATA_DIR = PROJECT_ROOT / "errata" / "v1.1"


@pytest.fixture(scope="module")
def manifest():
    path = ERRATA_DIR / "MANIFEST.json"
    if not path.exists():
        pytest.skip("errata set not built; run `python scripts/repair_items.py`")
    with open(path) as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def base_by_id():
    return {p["id"]: p for p in repair.load_corpus()}


# ──────────────────────────────────────────────────────────────
# The errata set as shipped
# ──────────────────────────────────────────────────────────────

def test_manifest_covers_exactly_the_flagged_items(manifest, base_by_id):
    flagged = {pid for pid, p in base_by_id.items() if repair.defects_of(p)}
    assert {e["id"] for e in manifest["items"]} == flagged


def test_every_replacement_is_defect_free():
    for path in sorted(ERRATA_DIR.glob("MURU-*.json")):
        with open(path) as handle:
            item = json.load(handle)
        item["_split"] = "errata"
        assert repair.defects_of(item) == [], path.name


def test_replacements_are_interchangeable_with_what_they_replace(base_by_id):
    for path in sorted(ERRATA_DIR.glob("MURU-*.json")):
        with open(path) as handle:
            item = json.load(handle)
        original = base_by_id[item["id"]]
        assert item["difficulty"] == original["difficulty"], item["id"]
        assert item["category"] == original["category"], item["id"]
        assert item["metadata"]["author"] == original["metadata"]["author"], item["id"]


def test_replacements_actually_differ_from_the_originals(base_by_id):
    for path in sorted(ERRATA_DIR.glob("MURU-*.json")):
        with open(path) as handle:
            item = json.load(handle)
        assert item["stem"] != base_by_id[item["id"]]["stem"], item["id"]


def test_every_replacement_carries_its_errata_provenance(manifest):
    for entry in manifest["items"]:
        with open(ERRATA_DIR / f"{entry['id']}.json") as handle:
            item = json.load(handle)
        errata = item["metadata"]["errata"]
        assert errata["version"] == "v1.1"
        assert errata["replaces"] == "v1.0"
        assert errata["defects"] == entry["defects"]
        assert set(errata["defects"]) <= {"D1", "D2", "D3"}


def test_base_corpus_is_untouched(base_by_id):
    """data/ must stay v1.0 — the whole point of shipping the fixes separately."""
    flagged = [pid for pid, p in base_by_id.items() if repair.defects_of(p)]
    assert len(flagged) == 280, (
        f"{len(flagged)} defective items in data/; the base corpus was expected to "
        "keep all 280 so the published numbers stay reproducible"
    )


def test_manifest_records_both_hashes(manifest):
    for entry in manifest["items"]:
        assert len(entry["sha256_v1_0"]) == 64
        assert len(entry["sha256_v1_1"]) == 64
        assert entry["sha256_v1_0"] != entry["sha256_v1_1"]


# ──────────────────────────────────────────────────────────────
# The regeneration itself
# ──────────────────────────────────────────────────────────────

def test_regeneration_is_deterministic(base_by_id):
    with open(PROJECT_ROOT / "problem_schema.json") as handle:
        schema = json.load(handle)
    by_author = repair.template_by_author()
    original = base_by_id["MURU-1258"]
    template = by_author[original["metadata"]["author"]]
    first, _ = repair.regenerate(original, template, schema)
    second, _ = repair.regenerate(original, template, schema)
    assert first == second


def test_regeneration_does_not_disturb_the_global_rng(base_by_id):
    """It reseeds internally, so it must leave the caller's stream alone."""
    with open(PROJECT_ROOT / "problem_schema.json") as handle:
        schema = json.load(handle)
    by_author = repair.template_by_author()
    original = base_by_id["MURU-1258"]
    template = by_author[original["metadata"]["author"]]

    random.seed(11)
    expected = [random.random() for _ in range(3)]
    random.seed(11)
    repair.regenerate(original, template, schema)
    assert [random.random() for _ in range(3)] == expected


def test_template_by_author_covers_every_generator_in_the_corpus(base_by_id):
    """`decision_payoff` signs its items `generator_decision`; the map is read
    off the generator rather than guessed, so a rename cannot silently orphan
    a template."""
    mapping = repair.template_by_author()
    authors = {
        p["metadata"]["author"]
        for p in base_by_id.values()
        if p["metadata"]["author"].startswith("generator_")
    }
    assert authors <= set(mapping), authors - set(mapping)


# ──────────────────────────────────────────────────────────────
# Indefinite articles
# ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("agriculture", "an"),
        ("archaeology", "an"),
        ("industrial engineering", "an"),
        ("automotive testing", "an"),
        ("urban planning", "an"),
        ("medical treatment", "a"),
        ("clinical", "a"),
        ("food manufacturing", "a"),
        # Vowel letter, consonant sound.
        ("university admissions", "a"),
        ("unit test", "a"),
        ("European market", "a"),
        # Consonant letter, vowel sound.
        ("hour-long trial", "an"),
        ("honest broker", "an"),
        # Acronyms are read letter by letter.
        ("ISP analysis", "an"),
        ("FDA review", "an"),
        ("MRI scan", "an"),
        ("CT scanner", "a"),
        ("GDP forecast", "a"),
    ],
)
def test_indefinite_article(phrase, expected):
    assert gp.indefinite_article(phrase) == expected


def test_with_article_capitalises_only_the_article():
    assert gp.with_article("agriculture", True) == "An agriculture"
    assert gp.with_article("agriculture") == "an agriculture"
    assert gp.with_article("medical treatment", True) == "A medical treatment"


# "Option A and Option B", "Treatment A is better", "Zone A after the update" —
# a bare label, not an article, and the only way to tell is what follows it.
_LABEL_FOLLOWERS = {
    "and", "is", "are", "was", "were", "or", "vs", "after", "over", "achieves",
    "against", "at", "accepts", "outperforms",
}


def test_generated_stems_agree_on_their_articles():
    """552 stems in v1.0 read 'A agriculture study' or 'A ISP analysis study'."""
    import re

    state = random.getstate()
    random.seed(2026)
    try:
        for name, template in sorted(gp.TEMPLATES.items()):
            low, high = template.difficulty_range
            for i in range(60):
                stem = template.generate(9000 + i, random.randint(low, high))["stem"]
                for match in re.finditer(r"\b[Aa]\s+([aeiouAEIOU][a-z]\w*)", stem):
                    word = match.group(1).lower()
                    if word in gp._TAKES_A or word in _LABEL_FOLLOWERS:
                        continue
                    pytest.fail(f"{name}: {match.group(0)!r} in {stem[:90]}")
    finally:
        random.setstate(state)


def test_the_article_sweep_would_catch_the_bug_it_was_written_for():
    """Guard the guard: the label exemption must not swallow a real mismatch."""
    import re

    stem = "A agriculture study measures the yield per hectare (tonnes)."
    hits = [
        m.group(0)
        for m in re.finditer(r"\b[Aa]\s+([aeiouAEIOU][a-z]\w*)", stem)
        if m.group(1).lower() not in gp._TAKES_A
        and m.group(1).lower() not in _LABEL_FOLLOWERS
    ]
    assert hits == ["A agriculture"]
