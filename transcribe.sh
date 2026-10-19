#!/usr/bin/env bash
# Compatibility entry point for MozhiFlow; all existing arguments remain supported.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment missing. Run: $ROOT/setup.sh" >&2
  exit 1
fi
exec "$PYTHON" "$ROOT/pipeline.py" "$@"
