#!/usr/bin/env bash
# run_prompt_v2_replication.sh — re-run the robustness subset under prompt v2.
#
# The archived panel was collected under prompt version 1, which asked for "a
# single number" without saying which unit it should be in; 24.7% of its
# recorded errors turned out to be right answers in another admissible unit
# (evaluation/unit_accounting.py). That correction is applied post hoc, so it
# needs an independent check: does a prompt that states the convention actually
# produce what the correction predicts?
#
# This runs the fixed 100-problem robustness subset (data/robustness_subset.json)
# under prompt v2 on the three endpoints the provider still serves. Arms land in
# evaluation/baselines/ablations/ and are never picked up as leaderboard rows.
#
# Compare with: python evaluation/compare_prompt_versions.py
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

KEY_FILE="evaluation/.groq_key"
[ -f "$KEY_FILE" ] || { echo "missing $KEY_FILE" >&2; exit 1; }
export GROQ_API_KEY
GROQ_API_KEY="$(cat "$KEY_FILE")"

PYTHON="${PYTHON:-.venv/bin/python}"
IDS="data/robustness_subset.json"
TAG="${TAG:-promptv2}"
LOG_DIR="evaluation/baselines/logs"
mkdir -p "$LOG_DIR"

# Slowest-and-most-informative first: if a daily token budget runs out midway,
# the leader's arm is the one already collected. Each model is independent, so
# a failure in one does not abort the others.
for model in gpt-oss-120b llama-3.3-70b llama-3.1-8b; do
    echo "=== $model  ($(date '+%H:%M:%S')) ==="
    PYTHONPATH= "$PYTHON" evaluation/run_eval.py \
        --model "$model" \
        --ids "$IDS" \
        --tag "$TAG" \
        --save 2>&1 | tee -a "$LOG_DIR/prompt_v2_${model}.log" | tail -5
    echo "--- $model done ($(date '+%H:%M:%S')) ---"
done

echo
echo "All arms attempted. Compare against the archived panel with:"
echo "  $PYTHON evaluation/compare_prompt_versions.py"
