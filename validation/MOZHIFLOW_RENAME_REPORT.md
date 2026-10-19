# MozhiFlow rename verification

## Result

The project now identifies as **MozhiFlow**, with distribution/package/repository name `mozhiflow` and console command `mozhiflow`. The package description is "Multilingual audio and video transcription, translation, and speaker-aware transcripts."

The local directory remains at its existing location. See [migration instructions](../docs/RENAMING.md) for the intended `~/tools/mozhiflow` path, checkpoint constraints, directory alias option, and manual GitHub steps. No remote operation or physical project move was performed. This working directory has no Git metadata; existing Git history and remote ownership could not be inspected.

## Source files changed or added

Every authored file in this change is listed below. Installation/build products under `.venv/`, `build/`, and `mozhiflow.egg-info/` are generated and ignored.

- Metadata and distribution: `pyproject.toml`, `MANIFEST.in`.
- Package: `mozhiflow/__init__.py`, `mozhiflow/__main__.py`, `mozhiflow/cli.py`, `mozhiflow/_worker.py`, `mozhiflow/pipeline.py`, `mozhiflow/audio_preprocess.py`, `mozhiflow/providers.py`, `mozhiflow/sarvam_provider.py`, `mozhiflow/ollama_provider.py`, `mozhiflow/toolkit_core.py`.
- Compatibility modules: `pipeline.py`, `audio_preprocess.py`, `providers.py`, `sarvam_provider.py`, `ollama_provider.py`, `toolkit_core.py`.
- Setup and wrapper: `setup.sh`, `transcribe.sh`.
- Portable configuration and authentication examples: `config.example.toml`, `config.hybrid.toml`, `.env.1password.example`.
- Ignore rules: `.gitignore`.
- Documentation: `README.md`, `HYBRID.md`, `original/README.md`, `docs/RENAMING.md`, `validation/MOZHIFLOW_RENAME_REPORT.md`.
- Validation tools: `scripts/validate_hybrid.py`, `scripts/validate_real.py`.
- Tests: `tests/test_pipeline.py`, `tests/test_preprocessing.py`, `tests/test_sarvam.py`, `tests/test_sarvam_dual.py`, `tests/test_ollama.py`, `tests/test_hybrid_integration.py`, `tests/test_diarization.py`, `tests/test_native_budget.py`, `tests/test_packaging.py`.

## Compatibility and preservation

The implementation moved into the package with relative internal imports. The former module names alias the same module objects, preserving exceptions, imports, and monkeypatch behavior. `transcribe.sh` and `python pipeline.py` remain supported; `python -m mozhiflow` is also available. The isolated MLX worker has a package-aware bootstrap that works outside the checkout.

Setup installs the package and its console command in the isolated environment. Existing valid environments are reused without reinstalling the managed Python interpreter. MLX remains optional; VAD is exposed as an optional package extra. The missing shareable 1Password example was added using placeholders.

No processing schema, source identity format, fingerprint input, paid-job token, output template, or reservation logic changed. Default configuration lookup remains at the source checkout for source/editable runs; independent wheel installations use the working directory. Explicit configuration paths continue to work.

SHA-256, size, and modification-time comparisons passed for **181 existing files** across `input/`, `Transcripts/`, `validation/private/`, `saaras-test/`, `models/`, plus `config.toml`, `config.diarized.toml`, and `AGENTS.md`. This includes saved recordings, manifests, remote job IDs, ledgers, checkpoints, and translation artifacts. The private 1Password file was neither read nor modified.

All 13 discovered pipeline/budget locks were unlocked at inspection. Saved remote states comprised 12 downloaded and 3 imported records. Process enumeration was denied by the execution environment, so an entirely idle machine could not be established. Existing absolute operational paths were retained.

## Verification

| Check | Result |
| --- | --- |
| Baseline `unittest` suite before changes | 129 tests passed |
| Existing suite after packaging/import changes | 129 tests passed |
| Final suite including package compatibility checks | 132 tests passed |
| Same suite from an extracted source distribution with a fresh `.venv` | 132 tests passed |
| `setup.sh --sarvam-only` in existing and fresh environments | Passed |
| `setup.sh` with existing MLX dependencies | Passed |
| Wheel and source distribution builds | Passed using installed setuptools backend |
| Independent wheel installation into another fresh environment | Passed |
| Clean-source/wheel CLI smoke checks | 18 passed |
| Shell syntax, setup help, Python compilation | Passed |
| `.gitignore` checks in an isolated temporary Git fixture | 16 private/generated paths ignored; 5 shareable paths included |
| Distribution inspection | No private configurations, credentials, recordings, job state, models, or environments included |
| Preserved data checks | 181 files unchanged |

Smoke checks exercised help, version, the compatibility wrapper, module invocation, validator help, legacy imports, and synthetic Unicode-named WAV dry runs. Wheel dry runs used only the wheel installation and a separate working-directory configuration; they did not reference the developer checkout or create output trees. The new worker bootstrap test runs outside the checkout and checks failure handling without loading optional inference dependencies.

Clean onboarding followed the README's setup, activation/command, configuration-copy, and input-directory workflow from an extracted distribution. Python 3.12.15, Homebrew, FFmpeg/FFprobe, and uv were already installed. Package builds used a fresh writable uv cache populated from the existing cached setuptools distribution. An initial online build attempt failed because PyPI DNS was unavailable; the normal isolated build/install path subsequently passed with `UV_OFFLINE=true` and that copied cache. Downloads of system tools, a new Python runtime, optional model weights, and uncached dependencies were not validated.

No live API calls, uploads, paid jobs, GPU inference, real VAD quality checks, or semantic translation-quality claims were made. Existing provider behavior is covered by the regression suite's synthetic media and mocked provider/model responses. Live provider and 1Password account checks remain dependent on the user's private environment.

Execution logs, smoke results, lock inventory, and the preservation baseline are in `/private/tmp/mozhiflow-rename/`; they are local temporary evidence, not committed artifacts.
