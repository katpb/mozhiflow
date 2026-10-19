#!/usr/bin/env bash
# Install an isolated MLX Whisper environment on Apple Silicon.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"

if [[ "$(uname -s)" != 'Darwin' || "$(uname -m)" != 'arm64' ]]; then
  echo 'This setup is intended for Apple Silicon macOS (arm64).' >&2
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

echo 'Installing uv-managed Python 3.12 (keeps pyenv and global Python unchanged)...'
uv python install 3.12
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 --managed-python "$VENV"
fi

"$VENV/bin/python" - <<'PY'
import hashlib
import platform
import sys
assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12, got {sys.version}"
assert platform.machine() == "arm64", f"Expected arm64, got {platform.machine()}"
assert hashlib.blake2b(b'check').hexdigest(), "BLAKE2 self-test failed"
print(f'Python health check OK: {sys.version.split()[0]} ({platform.machine()})')
PY

# Isolated venv install, never the global pip or an active project.
echo 'Installing mlx-whisper into the isolated environment...'
uv pip install --python "$VENV/bin/python" mlx-whisper
if [[ ! -x "$VENV/bin/mlx_whisper" ]]; then
  echo 'Install did not create mlx_whisper. Please inspect uv output above.' >&2
  exit 1
fi

echo
echo 'Installation successful.'
echo 'Run: ./transcribe.sh "/path/to/recording.mov"'
echo 'First transcription will download the MLX Whisper model from Hugging Face.'
