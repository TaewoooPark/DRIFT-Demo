#!/usr/bin/env bash
# DRIFT-Demo setup — vendor DRIFT (pinned), create the venv, install.
#
#   bash scripts/setup.sh                # clone DRIFT from GitHub at $DRIFT_REF
#   bash scripts/setup.sh /path/to/DRIFT # clone from a local checkout instead
#
# The demo instruments DRIFT internals (Orchestrator._prefill/_decode,
# Node._relay, TorchShardEngine.load/forward/head_argmax), so the vendored
# checkout is PINNED. To move to a newer DRIFT deliberately:
#   rm -rf vendor/DRIFT && DRIFT_REF=<tag-or-branch> bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DRIFT_SRC="${1:-https://github.com/TaewoooPark/DRIFT}"
DRIFT_REF="${DRIFT_REF:-v1.0.0}"

if [ ! -d vendor/DRIFT ]; then
  echo "[setup] cloning DRIFT @ $DRIFT_REF from $DRIFT_SRC …"
  git clone --depth 1 --branch "$DRIFT_REF" "$DRIFT_SRC" vendor/DRIFT
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
