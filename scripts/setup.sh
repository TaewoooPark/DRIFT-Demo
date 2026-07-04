#!/usr/bin/env bash
# DRIFT-Demo setup — vendor DRIFT, create the venv, install.
#
#   bash scripts/setup.sh                # clone DRIFT from GitHub
#   bash scripts/setup.sh /path/to/DRIFT # clone from a local checkout instead
set -euo pipefail
cd "$(dirname "$0")/.."

DRIFT_SRC="${1:-https://github.com/TaewoooPark/DRIFT}"

if [ ! -d vendor/DRIFT ]; then
  echo "[setup] cloning DRIFT from $DRIFT_SRC …"
  git clone "$DRIFT_SRC" vendor/DRIFT
else
  echo "[setup] vendor/DRIFT already present — leaving it as is"
fi

if [ ! -d .venv ]; then
  uv venv --python 3.12 .venv
fi
VIRTUAL_ENV="$PWD/.venv" uv pip install -e vendor/DRIFT

echo
echo "[setup] done. run the demo with:"
echo "    .venv/bin/python -m demo"
