#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/jupyter ]; then
  echo 'Najpierw uruchom: bash scripts/setup.sh'
  read -r -p 'Enter, aby zamknąć.'
  exit 1
fi
open index.html
exec .venv/bin/jupyter lab --ServerApp.ip=127.0.0.1 --ServerApp.root_dir="$PWD"
