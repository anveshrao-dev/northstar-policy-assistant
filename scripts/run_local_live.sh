#!/usr/bin/env bash
# Explicit Mac loopback launcher. No downloads, ledger creation or price changes.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
test "$(pwd -P)" = "$root" || { echo "Run from the Northstar Atlas checkout root" >&2; exit 1; }
export NORTHSTAR_PROJECT_ROOT="$root"
export PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -B -m northstar_atlas.local_live "$@"
