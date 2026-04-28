#!/usr/bin/env bash
# Dev server with auto-reload.
# Production runs under systemd (f1racecraft.service) — this is for local iteration.
#
# Usage:   ./scripts/serve.sh
# Override port:   PORT=8766 ./scripts/serve.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8765}"
VENV="${VENV:-/home/anurag/.virtualenvs/f1-strategist}"

if systemctl is-active --quiet f1racecraft.service 2>/dev/null; then
  echo "⚠️  systemd is serving f1racecraft on port 8765."
  echo "    Stop it first:    sudo systemctl stop f1racecraft.service"
  echo "    Or use another port:    PORT=8766 ./scripts/serve.sh"
  echo ""
fi

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "❌ Port ${PORT} is already in use. Pick another (PORT=8766) or stop the process."
  exit 1
fi

echo "→ Dev server: http://127.0.0.1:${PORT}  (auto-reload, Qwen3 preload skipped)"
F1_DEV_MODE=1 exec "${VENV}/bin/python" -m uvicorn server.app:app \
  --host 127.0.0.1 --port "${PORT}" --reload "$@"
