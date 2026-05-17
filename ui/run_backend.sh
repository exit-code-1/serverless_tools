#!/usr/bin/env bash
# Launch the Serverless Predictor backend.
# Bind to all interfaces so a frontend on another host can reach it.

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PY_BIN="${PYTHON:-python}"
exec "${PY_BIN}" -m uvicorn ui.backend.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level "${LOG_LEVEL}"
