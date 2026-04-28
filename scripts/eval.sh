#!/usr/bin/env bash
# Canonical eval: 6 scenarios × N seeds × {random, untrained, trained, expert}.
# Idempotent — skips if output JSON exists; pass FORCE=1 to re-run.
#
# Usage:   ./scripts/eval.sh
# Quick:   N_SEEDS=2 ./scripts/eval.sh
# Force:   FORCE=1 ./scripts/eval.sh
# Specific model:   MODEL=grpo_v3/merged ./scripts/eval.sh

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-/home/anurag/.virtualenvs/f1-strategist}"
MODEL="${MODEL:-grpo_v2/merged}"
OUT_JSON="${OUT_JSON:-results/eval_summary.json}"
OUT_PNG="${OUT_PNG:-results/eval_curve.png}"
N_SEEDS="${N_SEEDS:-5}"
FORCE="${FORCE:-0}"

if [[ "${FORCE}" != "1" && -f "${OUT_JSON}" ]]; then
  echo "✓ ${OUT_JSON} already exists. Use FORCE=1 to re-run."
  exit 0
fi

if [[ ! -d "${MODEL}" && "${MODEL}" != *"/"*"/"* ]]; then
  echo "❌ Model not found: ${MODEL}"
  echo "   Tried as local dir; for HF Hub use org/name format."
  exit 1
fi

mkdir -p "$(dirname "${OUT_JSON}")"
echo "→ Eval ${MODEL} · 6 scenarios × ${N_SEEDS} seeds × 4 modes"
exec "${VENV}/bin/python" evaluate.py \
  --modes random untrained trained expert \
  --model "${MODEL}" \
  --n-seeds "${N_SEEDS}" \
  --output-json "${OUT_JSON}" \
  --output-png "${OUT_PNG}" \
  "$@"
