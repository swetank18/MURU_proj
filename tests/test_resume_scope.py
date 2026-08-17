"""Tests for ``run_eval.load_prior_success`` scoping.

Resume exists so a run can be split across daily-token-budget windows. The
hazard is that an ablation arm answers the *same* problems as the published
panel under a *different* prompt: if resume drew on the panel, every subset
problem would already look answered, and the arm would either do nothing or
emit an archive stamped ``prompt_version: 2`` that is full of v1 answers.
These tests pin the scoping in both directions.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluation.run_eval as run_eval


def archive(path, model, pid, tag=None, prompt_version=2, ids_file=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "model": model,
            "run_tag": tag,
            "prompt_version": prompt_version,
            "problem_ids_file": ids_file,
            "raw_results": [{
                "problem_id": pid,
                "success": True,
                "parsed": {"point_estimate": 1.0},
            }],
        }, f)


def setup_root(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "PROJECT_ROOT", tmp_path)
    return tmp_path / "evaluation" / "baselines"


def test_untagged_resume_ignores_ablation_arms(tmp_path, monkeypatch):
    baselines = setup_root(tmp_path, monkeypatch)
    archive(baselines / "panel.json", "m", "MURU-0001")
    archive(baselines / "ablations" / "arm.json", "m", "MURU-0002", tag="promptv2")

    prior = run_eval.load_prior_success("m")

    assert set(prior) == {"MURU-0001"}


def test_tagged_resume_ignores_the_panel(tmp_path, monkeypatch):
    baselines = setup_root(tmp_path, monkeypatch)
    archive(baselines / "panel.json", "m", "MURU-0001")
    archive(baselines / "ablations" / "arm.json", "m", "MURU-0002",
            tag="promptv2", ids_file="data/robustness_subset.json")

    prior = run_eval.load_prior_success(
        "m", tag="promptv2", ids_file="data/robustness_subset.json"
    )

    assert set(prior) == {"MURU-0002"}


def test_tagged_resume_ignores_other_arms(tmp_path, monkeypatch):
    """A different tag, prompt version, or subset is a different experiment."""
    baselines = setup_root(tmp_path, monkeypatch)
    ids = "data/robustness_subset.json"
    archive(baselines / "ablations" / "same.json", "m", "MURU-0001",
            tag="promptv2", ids_file=ids)
    archive(baselines / "ablations" / "other_tag.json", "m", "MURU-0002",
            tag="temperature", ids_file=ids)
    archive(baselines / "ablations" / "other_prompt.json", "m", "MURU-0003",
            tag="promptv2", prompt_version=1, ids_file=ids)
    archive(baselines / "ablations" / "other_ids.json", "m", "MURU-0004",
            tag="promptv2", ids_file="data/other_subset.json")
    archive(baselines / "ablations" / "other_model.json", "n", "MURU-0005",
            tag="promptv2", ids_file=ids)

    prior = run_eval.load_prior_success("m", tag="promptv2", ids_file=ids)

    assert set(prior) == {"MURU-0001"}


def test_later_archive_wins_on_collision(tmp_path, monkeypatch):
    baselines = setup_root(tmp_path, monkeypatch)
    first = baselines / "ablations" / "a.json"
    second = baselines / "ablations" / "b.json"
    archive(first, "m", "MURU-0001", tag="t")
    archive(second, "m", "MURU-0001", tag="t")
    with open(second) as f:
        data = json.load(f)
    data["raw_results"][0]["parsed"]["point_estimate"] = 2.0
    with open(second, "w") as f:
        json.dump(data, f)
    import os
    os.utime(second, (first.stat().st_atime + 10, first.stat().st_mtime + 10))

    prior = run_eval.load_prior_success("m", tag="t")

    assert prior["MURU-0001"]["parsed"]["point_estimate"] == 2.0


def test_unparsed_answers_are_not_treated_as_answered(tmp_path, monkeypatch):
    """A failure must stay retryable — that is the point of resuming."""
    baselines = setup_root(tmp_path, monkeypatch)
    path = baselines / "ablations" / "arm.json"
    archive(path, "m", "MURU-0001", tag="t")
    with open(path) as f:
        data = json.load(f)
    data["raw_results"].append(
        {"problem_id": "MURU-0002", "success": True, "parsed": {"point_estimate": None}}
    )
    data["raw_results"].append(
        {"problem_id": "MURU-0003", "success": False, "parsed": None}
    )
    with open(path, "w") as f:
        json.dump(data, f)

    prior = run_eval.load_prior_success("m", tag="t")

    assert set(prior) == {"MURU-0001"}
