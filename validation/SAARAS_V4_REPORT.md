# Saaras V4 dual-mode implementation report — 2026-10-18

The existing application now supports the requested primary workflow: local video/audio preparation, optional conservative VAD, one consolidated lossless FLAC, and independent Saaras V4 Batch `codemix` / `translate` jobs over the same payload. The original CLI, internal stage names, checkpoints, manifest and output writers are retained. No separate application or competing transcription launcher was created.

**Live Batch API validation was not performed.** No private recording was uploaded, no new remote job was created, and no new API charge was incurred during this work. New 1–2-hour remote processing and recognition/translation quality therefore remain unverified against the live service. Local replay of previously successful real provider responses is explicitly distinguished below.

## Inspection and official API verification

Inspected configuration, source identities, manifests, stage fingerprints, partial-output protection, remote intent/state files, inference checkpoints, provider abstraction and all existing tests before changing behavior. Baseline: 79 tests passed.

Verified the current official documentation for Batch creation, upload links, start, polling, downloads, supported modes, formats, timestamps, duration limits and pricing. The implementation uses `POST /speech-to-text/job/v1` with `job_parameters.model="saaras:v4"` and the selected `mode`; it does not use the legacy translation model/endpoint. FLAC auto-detection is used without inventing a FLAC value for the optional codec enum. Normal eligible recordings use one uploaded file per mode. Over-two-hour prepared audio is rejected with the theoretical increased job count, not silently split or charged.

References: [Batch guide](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/batch-api), [formats/model](https://docs.sarvam.ai/api/getting-started/models/saaras), [initiate](https://docs.sarvam.ai/api-reference/speech-to-text/stt/job/initiate), [upload](https://docs.sarvam.ai/api-reference/speech-to-text/stt/job/upload), [start](https://docs.sarvam.ai/api-reference/speech-to-text/stt/job/start), [status](https://docs.sarvam.ai/api-reference/speech-to-text/stt/job/status), [download](https://docs.sarvam.ai/api-reference/speech-to-text/stt/job/download), [pricing](https://www.sarvam.ai/api-pricing).

## Architecture and safeguards

- Existing `transcript` stage invokes codemix; existing `translation` stage independently invokes translate on the same prepared file. The latter needs neither a codemix checkpoint nor Ollama. MLX and the older optional Ollama profile remain available.
- FFmpeg prepares mono speech-rate PCM and optionally compresses it losslessly to FLAC. VAD retains padded intervals and consolidates them into one file. Codec, sample rate, bytes, durations, removed intervals and mappings are recorded. Matching audio-only inputs bypass conversion when no normalization/VAD is needed; other audio inputs go directly to audio processing rather than video extraction.
- Preparation is reused within the run and from a validated cache. Scratch-WAV retention is configurable; upload payload/state/raw responses remain retained for safe retries. Originals are never cleanup targets.
- Modes have distinct job IDs, remote states and raw responses. Pre-mutation intent persists before create/start, read retries are bounded, ambiguous outcomes never trigger a blind replacement, and definitive credit/authentication rejection can be retried without falsely requiring an unknown job ID.
- Saved consent covers explicitly selected modes and the exact prepared audio/recognition parameters, including an interruption before the second job is created. Both modes' estimated costs are reserved before the first new paid job under a shared, locked cumulative budget ledger.
- Default `flag` timing policy retains untimed/unmappable text and flags missing original positions; it does not manufacture SRT precision. Independent source/English chunks are not artificially aligned.
- Authentication remains environment-only. `op run` injection is supported without a custom launcher, `.env.1password` is private/ignored, the example contains references only, and no secrets were read, changed or rotated. Installed `op run --help` confirms its environment-file/subprocess interface.

## Automated results

**97 tests pass**, including all 79 existing tests. See `unit-tests.txt`.

New coverage includes actual synthetic MOV extraction to FLAC, direct reuse of compatible audio, source preservation, non-default sample-rate mapping, mapping across multiple removed gaps, templates and collision preflight, CLI precedence, both SaaS modes sharing exactly one prepared file, independent failure/resume, pending translation without re-upload, interruption after remote create, saved consent across an output failure, combined budget rejection before any job, missing timestamps without fabricated cues, configurable environment-variable names without leakage, safe retry after rejected credits, the exact two-hour boundary with two jobs, and blocking any automatic additional-job workflow beyond that boundary.

Normal tests fake all provider/network inference. They never upload private audio or require real credits. VAD decisions remain mocked; real Silero speech recall was not evaluated. The two-hour boundary test uses mocked duration/provider behavior, not a live two-hour recording.

Python compilation, `bash -n setup.sh transcribe.sh`, `./setup.sh --help`, CLI help and real local dry-run commands passed. Setup's package-install branch was not rerun because the existing Python/FFmpeg environment was already usable; no new global Homebrew installs were performed.

## Local real-sample validation

Generated a 25-second MOV locally by pairing a small synthetic black video stream with the existing real Tamil–English `saaras-test/sample.wav`. This does not upload or modify the provided audio. All new media/evidence lives under ignored `validation/private/saaras-v4/`.

Validated the video through the regular CLI and the existing short-sample validator using the archived Saaras codemix and English responses, explicitly marked as imports:

```bash
./transcribe.sh --config validation/private/saaras-v4/replay.toml --dry-run
.venv/bin/python scripts/validate_hybrid.py \
  --config validation/private/saaras-v4/replay.toml \
  --report validation/private/saaras-v4/evidence.json
./transcribe.sh --config validation/private/saaras-v4/replay.toml
```

Observed results:

- Original/retained duration: 25.0 seconds; removed duration: 0.0 seconds with VAD off.
- Shared output: 16 kHz mono FLAC, 409,325 bytes.
- Decoding the extracted FLAC back to PCM produced bytes identical to the original WAV speech samples: no extra codec loss.
- Source MOV SHA-256, size and modification time matched the captured pre-processing identity.
- Both sets of TXT/SRT/JSON were nonempty. Each archived response has one 0–25-second timestamped chunk; both mapped correctly.
- Full text matches the corresponding archived provider transcript exactly, raw response bytes are preserved, and output hashes are intact.
- Translation-only validation retry skipped completed work; a subsequent full CLI rerun reported `0 completed, 2 skipped, 0 failed`.
- Import provenance records zero new uploaded duration/cost and no live remote job ID. No new translation was generated by the provider in this test.

Evidence: local `dry-run.json`, `public-config-dry-run.json`, `evidence.json`, `source-before.json`, `local-inspection.json`, generated artifacts and manifest under `validation/private/saaras-v4/`.

The archived provider texts and generated outputs were inspected for script/text retention, numeric content and timestamp shape; they were not newly judged against the full original conversation for semantic accuracy. The replay proves application conversion/output/retry handling, not live V4 Batch compatibility or fresh transcription quality.

## Files changed

Existing files updated:

- `pipeline.py`: configuration/CLI, mode naming, output directories/collision detection, shared preparation, consent/budget orchestration and dry-run observability.
- `audio_preprocess.py`: FLAC, configurable rate, direct-audio handling, explicit removed intervals, codec/size metadata and retention.
- `sarvam_provider.py`: independent modes, shared audio formats, environment-name configuration, per-mode reconciliation, combined reservations, definitive-vs-ambiguous failures and timing flags.
- `providers.py`: Sarvam audio-translation adapter; retain other providers and avoid Ollama preflight for the primary path.
- `setup.sh`: optional `--sarvam-only` environment setup, while preserving default MLX installation.
- `config.toml`: requested local paths and primary two-mode settings.
- `.gitignore`: private credentials/config, source media, outputs/intermediates/state/caches and case variants.
- `README.md`: complete setup/security/configuration/usage/retry/limits/developer instructions.
- `HYBRID.md`: identify the older Ollama profile as optional, not the primary workflow.
- `scripts/validate_hybrid.py`: extend the existing explicit short-sample utility for independent Saaras modes, new output layout and optional reference comparison.
- `scripts/validate_real.py`: reuse the shared output-directory resolver, preserving both old and new layouts.
- `validation/unit-tests.txt`: latest test evidence.

New repository files:

- `config.example.toml`: portable working primary configuration.
- `.env.1password.example`: placeholder secret-reference template, no actual key.
- `tests/test_sarvam_dual.py`: additional mocked integration and local media/configuration regressions.
- `validation/SAARAS_V4_REPORT.md`: this report.

The requested `input/` directory was created without moving, renaming or deleting existing recordings in `inputs/`. Existing results, MLX model files and private secret values were left untouched. No Git repository was initialized and nothing was published or pushed.

## Exact application commands and remaining validation

After adding recordings under `input/`, supplying your own key/reference and reviewing the dry-run/budget, explicitly authorize initial cloud work:

```bash
op run --env-file=.env.1password -- \
  ./transcribe.sh --config config.toml --allow-upload
```

For the same authorized work, the expected everyday resume/skip command is:

```bash
op run --env-file=.env.1password -- \
  ./transcribe.sh --config config.toml
```

Non-1Password users can run `./transcribe.sh --config config.toml` in an already securely injected environment, adding `--allow-upload` for new work. English-only retries use `--only translation` and do not repeat successful codemix processing.

Outstanding: an explicitly authorized live short Batch test with the developer's own credentials/credits, then representative long-recording performance and semantic review. Real provider chunk timing/format edge cases, current account pricing and service errors cannot be fully established by mocks/replay. API estimates exclude taxes and are not provider-enforced spending caps. Ambiguous create/start outcomes may require dashboard reconciliation rather than automatic progress. Quiet-speech VAD recall and translation preservation of names, numbers, negation/conditions remain review requirements. No claim of verified real transcription/translation quality is made.
