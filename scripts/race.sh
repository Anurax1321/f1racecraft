#!/usr/bin/env bash
# Run one race end-to-end. The fastest sanity check.
#
# Usage:                ./scripts/race.sh
# Different scenario:   TASK=late_safety_car ./scripts/race.sh
# Different seed:       SEED=42 ./scripts/race.sh
# With trained model:   MODEL=grpo_v2/merged ./scripts/race.sh

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-/home/anurag/.virtualenvs/f1-strategist}"
TASK="${TASK:-weather_roulette}"
SEED="${SEED:-7}"
MODEL="${MODEL:-heuristic}"

echo "→ Race · ${TASK} · seed=${SEED} · policy=${MODEL}"
exec "${VENV}/bin/python" inference.py \
  --task "${TASK}" --seed "${SEED}" --model "${MODEL}" --n-episodes 1 "$@"
