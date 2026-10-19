# Speaker diarization validation — 2026-10-19

Implemented in the existing application; no launch script or 1Password integration changes.

## Configuration and outputs

- Existing `sarvam.diarization` remains supported; `with_diarization` is an equivalent TOML alias. Conflicting values are rejected. The API receives `with_diarization` alongside existing `with_timestamps`.
- Optional `num_speakers` accepts integers 1–20. Omission sends no count and leaves detection to Sarvam. A count requires diarization; speaker-attributed outputs require timestamps.
- Optional `speaker_names` maps `codemix` and `translate` separately to provider-ID/name dictionaries. Personal names are user assertions and should be assigned only after inspecting the relevant recording/job.
- TXT includes speaker labels and original timestamp ranges. SRT includes speaker labels on valid cues. JSON preserves full provider transcript, entry text, raw provider IDs, optional names, speaker status, original/processed times, gap-spanning original intervals, and independent mode/job namespaces. Unknown/explicitly uncertain speakers receive no guessed identity. Raw provider bytes and checksums remain retained.
- Existing VAD mapping is reused; independently generated modes are not aligned and their speaker IDs are not equated.

## Paid-work safeguards

Changing diarization or speaker count for successful/submitted work stops preflight before budget reservation or new submission, including with `--overwrite`. This includes failed remote jobs, ambiguous creates, and recoverable remote state when the latest manifest lacks provider metadata. Restore original settings to recover those jobs. Deliberately new diarized work requires a separately selected output location, preview, sufficient scoped budget, credentials and explicit upload authorization.

Speaker-name changes affect rendering, not paid inference tokens. With intact raw responses, explicit `--overwrite` regenerates named outputs without network calls. Existing successful outputs otherwise remain unchanged. Manifests retain original stage settings and provider IDs.

## Pricing verification

Official pages disagree on 2026-10-19:

- API docs: ₹45/hour (₹0.75/minute), including translated diarization: https://docs.sarvam.ai/api/getting-started/pricing
- Website pay-as-you-go Batch diarization: ₹0.59/minute: https://www.sarvam.ai/api-pricing

The conservative configurable default is now ₹0.75/minute in Python and the machine/example/hybrid configurations. Non-diarized ₹0.49/minute configuration is unchanged. Whole-minute estimation and default margin 1.2 remain local safeguards: both diarized modes estimate ₹108 for 60 minutes or ₹216 for 120 minutes, excluding tax. Both modes/files enter the existing scoped budget preflight; no cap/history reset occurs. Verify account-specific pricing before paid work.

Official request/response reference: https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/how-to/enable-speaker-diarization

## Verification

- Full suite: **129 tests passed**, including all previous 115 tests and 14 new mocked diarization tests. Command: `.venv/bin/python -m unittest discover -s tests`. Output: `validation/unit-tests.txt`.
- New coverage includes request parameters, automatic count omission, both modes, independent/same-valued IDs, named/unknown/uncertain speakers, timestamped TXT/SRT/JSON, VAD gap mapping, malformed/missing diarization/timestamps, raw replay, speaker-name changes without network, no paid replacement after configuration changes, count/type/alias validation, combined budget blocking and updated estimates.
- All three configuration files load and validate. Current machine configuration keeps diarization disabled, preserving existing jobs; it was not silently enabled.
- Current configured `--dry-run` succeeded locally without uploading audio. Hashes of the existing budget ledger and six remote state files remained unchanged.
- No live Sarvam request, recording upload, GitHub push, or real diarization-quality assessment was performed. Speaker clustering accuracy, overlapping speech, Tamil-English recognition and English translation fidelity remain unverified against new live provider output. Mock tests verify behavior and accounting, not model quality. Pricing inconsistency remains an external limitation.

## Files changed

`pipeline.py`, `sarvam_provider.py`, `config.toml`, `config.example.toml`, `config.hybrid.toml`, `README.md`, `tests/test_diarization.py`, `validation/unit-tests.txt`, and this report.

To enable for deliberately new work, change the existing `diarization=false` line to `true`; optionally set `num_speakers=2`. Use existing `./transcribe.sh --config config.toml --dry-run` first. Existing completed non-diarized recordings require an explicitly selected separate output directory before requesting new paid diarization. README documents the naming tables and replay behavior. The CLI and secure `op run --env-file=.env.1password -- ...` workflow are unchanged.
