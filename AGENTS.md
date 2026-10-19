# Repository Guidelines

## Project Structure & Module Organization

This macOS toolkit transcribes Tamil–English recordings and produces English translations. `transcribe.sh` launches `pipeline.py`, which handles configuration, orchestration, checkpoints, and output generation. `audio_preprocess.py` handles media preparation; `providers.py`, `sarvam_provider.py`, and `ollama_provider.py` implement backend integrations. Shared utilities live in `toolkit_core.py`.

Regression tests live in `tests/`; validation utilities live in `scripts/`, with reports and sample configurations under `validation/`. `README.md` documents the primary Saaras workflow; `HYBRID.md` covers the optional Ollama workflow. Keep local recordings in `input/`, generated results in `Transcripts/`, and model weights in `models/`. `original/` preserves earlier scripts.

## Build, Test, and Development Commands

- `./setup.sh --sarvam-only`: create the isolated Python 3.12 environment and install required system tools through Homebrew/uv.
- `./setup.sh`: additionally install MLX Whisper for Apple Silicon.
- `uv pip install --python .venv/bin/python -r requirements-hybrid.txt`: install optional hybrid/VAD dependencies.
- `./transcribe.sh --help`: inspect supported CLI options.
- `./transcribe.sh --config config.toml --dry-run "input/meeting.mov"`: inspect media, configuration, and budget without API charges.
- `.venv/bin/python -m unittest discover -s tests -v`: run the regression suite.

There is no separate compilation step.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Use type annotations and `pathlib.Path` where consistent with nearby code. Keep shell wrappers compatible with Bash and retain `set -euo pipefail`. No formatter or linter configuration is currently supplied.

## Testing Guidelines

Use standard-library `unittest` and `unittest.mock`; name files `tests/test_*.py` and methods `test_*`. Cover configuration validation, timestamp mapping, resumability, and budget/consent behavior when changing those paths. Mock provider/model calls and use temporary directories or synthetic media. FFmpeg-dependent checks require FFmpeg/FFprobe. No numeric coverage threshold is configured; automated tests do not establish transcription quality.

## Commit & Pull Request Guidelines

This checkout has no Git metadata, so existing commit conventions cannot be verified. Use concise imperative subjects, such as `Fix timestamp mapping after silence removal`. PRs should describe behavior changes, list validation commands/results, link relevant issues, and update documentation or example TOML when options change.

## Security & Configuration

Copy `config.example.toml` for local settings. Inject `SARVAM_API_KEY` through the environment or 1Password; never commit credentials, recordings, or private transcripts. Cloud uploads require explicit `--allow-upload` consent and budget review. Preserve original media, provider checkpoints, and budget ledgers; deleting state can cause duplicate paid jobs.
