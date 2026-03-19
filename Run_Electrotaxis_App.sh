#!/usr/bin/env bash
set -euo pipefail

# Launcher (macOS/Linux):
# - Ensures venv + deps exist (runs Setup_Electrotaxis.sh if needed)
# - Opens browser
# - Runs Streamlit

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG="$ROOT/launcher.log"
{
  echo
  echo "===== $(date) RUN (mac/linux) ====="
} >> "$LOG"

PORT="${PORT:-8501}"
PY="$ROOT/.venv-cyto2/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "venv missing; running setup..." | tee -a "$LOG"
  bash "$ROOT/Setup_Electrotaxis.sh"
fi

if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
  echo "streamlit missing/broken; running setup..." | tee -a "$LOG"
  bash "$ROOT/Setup_Electrotaxis.sh"
fi

# Open browser (best-effort)
if command -v open >/dev/null 2>&1; then
  open "http://localhost:${PORT}" >/dev/null 2>&1 || true
fi

echo "Starting Streamlit on http://localhost:${PORT}" | tee -a "$LOG"

# Keep logs visible + saved.
"$PY" -m streamlit run "$ROOT/streamlit_app.py" --server.address localhost --server.port "$PORT" 2>&1 | tee -a "$LOG"

