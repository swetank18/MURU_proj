#!/usr/bin/env python3
"""
salvage_from_log.py — Recover partial real-LLM results from a run_eval.py log.

When a run_eval.py invocation hits a daily-token-budget exhaustion mid-run,
the JSON results file is never written. The progress log, however, contains
one line per scored problem with the parsed point estimate and confidence.
This script parses that log and reconstructs a partial results JSON suitable
for downstream aggregation (paper tables, leaderboard).

Salvaged predictions lack the predicted_framework and predicted_interval
fields (those are not echoed to the log), so framework-match rate cannot be
computed for the salvaged subset. The other three headline metrics
(Accuracy@CI, ECE, overconfidence) are recoverable in full.

Usage:
    python evaluation/salvage_from_log.py \
        --log evaluation/baselines/logs/gpt-oss-120b.log \
        --model gpt-oss-120b
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import MURUMetrics, Prediction


LINE_RE = re.compile(
    r"\[(\d+)/(\d+)\]\s+(MURU-\d+)\s+\.\.\.\s+✓\s+\(est=([-+]?\d*\.?\d+),\s+conf=([-+]?\d*\.?\d+)\)"
)


def load_problems(test_dir: Path) -> dict:
    problems = {}
    for f in sorted(test_dir.rglob("MURU-*.json")):
        with open(f) as fh:
            p = json.load(fh)
            problems[p["id"]] = p
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to run_eval.py log file")
    ap.add_argument("--model", required=True, help="Model identifier")
    ap.add_argument("--out", help="Output JSON path (default: auto)")
    args = ap.parse_args()

    log_path = Path(args.log)
    text = log_path.read_text()

    matches = LINE_RE.findall(text)
    if not matches:
        print(f"No successful problem lines found in {log_path}", file=sys.stderr)
        sys.exit(1)

    test_dir = PROJECT_ROOT / "data" / "test"
    problems_by_id = load_problems(test_dir)

    raw_results = []
    predictions = []
    seen = set()
    for _idx, _total, pid, est, conf in matches:
        if pid in seen:
            continue
        seen.add(pid)
        if pid not in problems_by_id:
            continue
        est_f = float(est)
        conf_f = float(conf)
        raw_results.append({
            "problem_id": pid,
            "response": "[salvaged from log; full response not available]",
            "parsed": {
                "framework": None,
                "point_estimate": est_f,
                "confidence_interval": None,
                "confidence": conf_f,
            },
            "success": True,
            "salvaged": True,
        })
        predictions.append(Prediction(
            problem_id=pid,
            predicted_answer=est_f,
            predicted_confidence=conf_f,
            predicted_interval=None,
            predicted_framework=None,
            raw_response="",
        ))

    if not predictions:
        print("No valid predictions recovered.", file=sys.stderr)
        sys.exit(1)

    # Compute metrics on the answered subset.
    sub_problems = [problems_by_id[p.problem_id] for p in predictions]
    metrics = MURUMetrics(sub_problems, predictions)
    summary = metrics.compute_all()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_").replace(".", "_")
    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "evaluation" / "baselines" / f"{model_slug}_{timestamp}_salvaged.json"
    )

    data = {
        "model": args.model,
        "timestamp": timestamp,
        "n_problems": len(problems_by_id),
        "n_predictions": len(predictions),
        "salvaged": True,
        "salvage_note": (
            "Reconstructed from run_eval.py log lines. predicted_framework and "
            "predicted_interval are unavailable; framework-match metric should "
            "be treated as missing for this entry."
        ),
        "metrics": summary,
        "raw_results": raw_results,
    }
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"Saved {len(predictions)} salvaged predictions to {out.relative_to(PROJECT_ROOT)}")
    print(f"Accuracy: {summary['accuracy']:.3f}")
    print(f"ECE: {summary['ece']:.3f}")
    print(f"Overconfidence: {summary['overconfidence_rate']:.3f}")


if __name__ == "__main__":
    main()
