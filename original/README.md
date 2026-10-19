> Historical implementation retained for reference. Use MozhiFlow and the setup instructions in the root README for current installs.

# Local Tamil transcription for Apple Silicon macOS

A reusable, no-subscription transcription kit based on Apple's MLX Whisper, with a dedicated uv-managed Python environment. Works with MOV, MP4, M4A, MP3, WAV, AAC, FLAC, MKV and WebM recordings.

## Prerequisites

- Apple Silicon Mac (arm64), Homebrew installed.
- Internet for the **initial** Python/package/model downloads. Later, processing occurs locally.
- Enough free space for MLX model weights (about **3.1 GB** for Large V3 or **1.6 GB** for Turbo), package dependencies, and outputs.
- `ffmpeg` is installed automatically by setup if absent.

## Installation

1. Extract this folder somewhere permanent, e.g. `~/tools/mozhiflow`.
2. Open Terminal and run:

```bash
cd ~/tools/mozhiflow
chmod +x setup.sh transcribe.sh
./setup.sh
```

This creates `./.venv` with uv-managed Python 3.12 and `mlx-whisper`. It does not modify your active `pyenv` Python or install into other projects.

## First transcription

```bash
./transcribe.sh "$HOME/Movies/2026-10-06 14-11-21(1).mov"
```

Whisper downloads the Large V3 MLX model on first use, then transcribes locally. File paths **with spaces and parentheses** must be quoted.

Default output location:

```text
~/Transcripts/
  2026-10-06 14-11-21(1)/
    2026-10-06 14-11-21(1).txt
    2026-10-06 14-11-21(1).srt
    2026-10-06 14-11-21(1).vtt
    2026-10-06 14-11-21(1).tsv
    2026-10-06 14-11-21(1).json
```

The transcript remains in the original language because the task is `transcribe`, not `translate`.

## Common options

```bash
# Faster multilingual model, with possible loss of recognition quality
./transcribe.sh --fast "$HOME/Movies/meeting.mov"

# Batch transcribe a directory (recursively)
./transcribe.sh "$HOME/Movies/Meetings"

# English recording
./transcribe.sh --language en "$HOME/Movies/meeting.mp4"

# Auto-detect dominant spoken language
./transcribe.sh --language auto "$HOME/Movies/meeting.mp4"

# Include vocabulary context, useful for company/product names
./transcribe.sh --prompt 'Samsara, Kafka, API, Redis, integration' "$HOME/Movies/meeting.mov"

# Output to a different directory, overwriting existing transcript
./transcribe.sh --output-dir "$HOME/Documents/MeetingTranscripts" --force "$HOME/Movies/meeting.mov"

# Test command construction without installing models
./transcribe.sh --dry-run "$HOME/Movies/meeting.mov"
```

## Accuracy notes

- `--language ta` fixes the dominant language as Tamil. Code-switching into English may still produce mistakes; review technical jargon and important decisions.
- For mixed meetings, try a full run with `--language ta` first. If dominant language varies widely, compare against `--language auto --force` (save outputs separately with `--output-dir` to avoid overwriting the first result).
- Models may hallucinate speech during silence/noise; review against timestamps, especially key requirements.
- Whisper does **not** identify speakers automatically. Speaker labeling requires separate diarization tooling or manual annotation.
- A prompt provides context, but is **not** a command to translate or enforce perfect jargon spellings.

## Troubleshooting

- `mlx_whisper` missing: rerun `./setup.sh`.
- Model download blocked: ensure access to huggingface.co for the first run.
- Out of memory: use `--fast` or smaller model `--model mlx-community/whisper-small-mlx`.
- `pyenv` BLAKE2 errors: these are in your separate Python 3.13 build. Setup uses a different Python managed by `uv` and tests `hashlib.blake2b` before installing.
- Audio issues: inspect with `ffprobe -hide_banner "file.mov"`. If needed, extract audio first: `ffmpeg -i "file.mov" -vn -ac 1 -ar 16000 "file.wav"`.
- Source recordings are not uploaded by this kit; dependency and model download contact third-party distribution servers. Keep sensitive source files and output transcripts in appropriately secured folders and check usage authorization.

Official upstream docs: https://github.com/ml-explore/mlx-examples/tree/main/whisper

## About versioning

This kit intentionally installs the current compatible `mlx-whisper` instead of fixing an unverified version. For strict repeatability, generate a local version snapshot after setup using:

```bash
uv pip freeze --python ./.venv/bin/python > requirements.lock.txt
```
