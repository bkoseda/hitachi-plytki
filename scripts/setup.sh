#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
if [ ! -x .venv/bin/python ]; then
  PING_PYTHON=""
  for candidate in python3.12 python3.11 python3 "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"; do
    if "$candidate" -c 'import sys; assert (3,11)<=sys.version_info<(3,14)' >/dev/null 2>&1; then
      PING_PYTHON="$candidate"
      break
    fi
  done
  if [ -z "$PING_PYTHON" ]; then
    echo 'Potrzebny Python 3.11–3.13 (zalecany 3.12). Zainstaluj go i ponów skrypt.' >&2
    exit 1
  fi
  "$PING_PYTHON" -m venv .venv
fi
if [ "${1:-}" = '--locked' ]; then
  .venv/bin/python -m pip install -r requirements-macos-py312.lock
  .venv/bin/python -m pip install --no-deps -e .
else
  .venv/bin/python -m pip install -e '.[embeddings]'
fi
.venv/bin/python -m ipykernel install --user --name pinglab --display-name 'Ping Lab (Python)'
.venv/bin/pinglab demo
.venv/bin/python scripts/warm_model.py
.venv/bin/python -m pip check
printf '\nGotowe. Uruchom Start.command albo .venv/bin/jupyter lab\n'
