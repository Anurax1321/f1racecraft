#!/usr/bin/env bash
# Training wrapper. Resumable via TRL's checkpoint mechanism.
#
# Usage:   ./scripts/train.sh [smoke|grpo]
#   smoke  — CPU dry-run (no GPU; exercises the loop)
#   grpo   — full GRPO on Qwen3-4B + LoRA (RTX 5090, ~30 min for 500 steps)
#
# Override checkpoint dir:    OUTPUT_DIR=grpo_v3 ./scripts/train.sh grpo
# Override step count:        MAX_STEPS=200 ./scripts/train.sh grpo

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-/home/anurag/.virtualenvs/f1-strategist}"
MODE="${1:-smoke}"

case "${MODE}" in
  smoke)
    echo "→ Smoke run · local-smoke backend · no GPU"
    exec "${VENV}/bin/python" train.py --backend local-smoke \
      --model heuristic --task multi --max-steps 12 \
      --output-dir grpo_smoke "${@:2}"
    ;;

  grpo)
    OUTPUT_DIR="${OUTPUT_DIR:-grpo_v3}"
    MAX_STEPS="${MAX_STEPS:-500}"
    BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"

    if [[ -d "${OUTPUT_DIR}" ]]; then
      latest=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
      if [[ -n "${latest}" ]]; then
        echo "✓ Found ${latest} — TRL will resume from there."
      fi
    fi

    echo "→ GRPO · ${BASE_MODEL} · ${MAX_STEPS} steps · output ${OUTPUT_DIR}"
    exec "${VENV}/bin/python" train.py --backend trl \
      --model "${BASE_MODEL}" --task multi \
      --max-steps "${MAX_STEPS}" \
      --batch-size 1 --grad-accum 32 \
      --output-dir "${OUTPUT_DIR}" "${@:2}"
    ;;

  *)
    echo "Usage: $0 [smoke|grpo]" >&2
    exit 1
    ;;
esac
