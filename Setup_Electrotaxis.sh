#!/usr/bin/env bash
set -euo pipefail

# One-time setup helper (macOS/Linux):
# - Creates .venv-cyto2
# - Installs requirements (mac uses requirements_mac.txt)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG="$ROOT/launcher.log"
{
  echo
  echo "===== $(date) SETUP (mac/linux) ====="
} >> "$LOG"

REQ_FILE="$ROOT/requirements.txt"
if [[ "$(uname -s)" == "Darwin" ]]; then
  if [[ -f "$ROOT/requirements_mac.txt" ]]; then
    REQ_FILE="$ROOT/requirements_mac.txt"
  fi
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "ERROR: Missing requirements file: $REQ_FILE" | tee -a "$LOG" >&2
  exit 1
fi

# Try python3.11 first, then python3.10, then python3
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_CMD="python3.11"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_CMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
else
  echo "ERROR: python3 not found. Install Python 3.10+ and try again." | tee -a "$LOG" >&2
  exit 1
fi

# Log detected version (helps debug mac installs)
PY_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $PY_VERSION at $PYTHON_CMD" | tee -a "$LOG"

if [[ ! -x "$ROOT/.venv-cyto2/bin/python" ]]; then
  echo "Creating venv at .venv-cyto2..." | tee -a "$LOG"
  "$PYTHON_CMD" -m venv "$ROOT/.venv-cyto2" 2>&1 | tee -a "$LOG"
fi

echo "Installing requirements (this may take a while)..." | tee -a "$LOG"
"$ROOT/.venv-cyto2/bin/python" -m pip install -r "$REQ_FILE" 2>&1 | tee -a "$LOG"

echo "Setup complete." | tee -a "$LOG"

