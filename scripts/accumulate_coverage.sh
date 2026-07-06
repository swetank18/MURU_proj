#!/usr/bin/env bash
#
# accumulate_coverage.sh — drive MURU-BENCH real-LLM evaluations toward full
# 301-problem coverage across several days.
#
# The Groq free tier enforces a hard tokens-per-day (TPD) cap, so a large
# model can only answer a few hundred problems per calendar day before the
# API starts returning TPD errors. This script runs `run_eval.py --resume`
# for each still-incomplete model: it skips problems already answered in
# prior archives, attempts the remainder until the daily budget is spent,
# merges everything into a fresh union archive, and re-aggregates the
# leaderboard. Run it once per day (via cron) and coverage climbs until the
# models are complete, at which point each run is a no-op.
#
# Key handling: the script reads GROQ_API_KEY from the environment, or from
# a gitignored key file if present (evaluation/.groq_key). Never commit the
# key. Rotate it at https://console.groq.com if it has ever been exposed.
#
# Usage:
#   scripts/accumulate_coverage.sh                 # default target models
#   scripts/accumulate_coverage.sh gpt-oss-120b    # explicit model list
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

PYTHON="$REPO/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

# --- API key ------------------------------------------------------------
if [ -z "${GROQ_API_KEY:-}" ] && [ -f "$REPO/evaluation/.groq_key" ]; then
    GROQ_API_KEY="$(tr -d ' \t\r\n' < "$REPO/evaluation/.groq_key")"
    export GROQ_API_KEY
fi
if [ -z "${GROQ_API_KEY:-}" ]; then
    echo "ERROR: GROQ_API_KEY not set and no evaluation/.groq_key file found." >&2
    exit 2
fi

# --- target models (large models that get TPD-capped) -------------------
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(gpt-oss-120b llama-3.3-70b)
fi

STAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
echo "  MURU-BENCH coverage accumulation — $STAMP"
echo "  Models: ${MODELS[*]}"
echo "════════════════════════════════════════════════════════════"

for m in "${MODELS[@]}"; do
    echo
    echo "----- $m -----"
    "$PYTHON" evaluation/run_eval.py --model "$m" --resume --save --delay 0.5
done

echo
echo "----- re-aggregating leaderboard -----"
"$PYTHON" evaluation/aggregate_real_llm.py

# --- report current union coverage so the log shows daily progress ------
"$PYTHON" - "${MODELS[@]}" <<'PY'
import sys
import evaluation.run_eval as r
print("\nUnion coverage after this run:")
for model in sys.argv[1:]:
    n = len(r.load_prior_success(model))
    flag = "  <-- COMPLETE" if n >= 301 else ""
    print(f"  {model:16s} {n:3d}/301{flag}")
PY
