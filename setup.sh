#!/usr/bin/env bash
# Install MozhiFlow and its isolated Python/FFmpeg environment, optionally with MLX Whisper.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
WITH_MLX=1
case "${1:-}" in
  --sarvam-only) WITH_MLX=0 ;;
  --help|-h) echo 'Usage: ./setup.sh [--sarvam-only]'; exit 0 ;;
  '') ;;
  *) echo 'Usage: ./setup.sh [--sarvam-only]' >&2; exit 2 ;;
esac

if [[ "$(uname -s)" != 'Darwin' ]]; then
  echo 'This setup script requires macOS.' >&2
  exit 1
fi
if (( WITH_MLX )) && [[ "$(uname -m)" != 'arm64' ]]; then
  echo 'MLX requires Apple Silicon. Use --sarvam-only for an Intel Mac.' >&2
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  echo 'Homebrew is required. Install it from https://brew.sh and rerun setup.sh.' >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo 'Installing FFmpeg with Homebrew...'
  brew install ffmpeg
else
  echo 'FFmpeg already installed.'
fi
if ! command -v uv >/dev/null 2>&1; then
  echo 'Installing uv with Homebrew...'
  brew install uv
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo 'Installing uv-managed Python 3.12 (keeps pyenv and global Python unchanged)...'
  uv python install 3.12
  uv venv --python 3.12 --managed-python "$VENV"
fi

"$VENV/bin/python" - "$WITH_MLX" <<'PY'
import hashlib
import platform
import sys
assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12, got {sys.version}"
if sys.argv[1] == "1":
    assert platform.machine() == "arm64", f"Expected arm64, got {platform.machine()}"
assert hashlib.blake2b(b'check').hexdigest(), "BLAKE2 self-test failed"
print(f'Python health check OK: {sys.version.split()[0]} ({platform.machine()})')
PY

# Install the primary command into the isolated environment.
uv pip install --python "$VENV/bin/python" --editable "$ROOT"

# Isolated venv install, never the global pip or an active project.
if (( WITH_MLX )); then
  echo 'Installing mlx-whisper into the isolated environment...'
  uv pip install --python "$VENV/bin/python" mlx-whisper
  if [[ ! -x "$VENV/bin/mlx_whisper" ]]; then
    echo 'Install did not create mlx_whisper. Please inspect uv output above.' >&2
    exit 1
  fi
fi
echo 'Installation successful.'
echo 'Run: source .venv/bin/activate'
echo 'Then: mozhiflow --help'
echo 'Compatibility: ./transcribe.sh remains available.'
if (( WITH_MLX )); then
  echo 'MLX weights download only when that backend is selected with --no-offline.'
fi
echo 'For Saaras: configure your environment/1Password key, dry-run, then authorize the initial upload.'
