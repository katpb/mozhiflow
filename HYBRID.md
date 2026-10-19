# MozhiFlow — Optional Saaras recognition and local Ollama translation

The `mozhiflow` CLI (also available as `transcribe.sh`) and stage runner now support two recognition providers (`mlx_whisper`, `sarvam`) and two translation providers (`mlx_whisper`, `ollama`). The primary `config.toml` now selects Saaras for both audio modes; see [README.md](README.md) for that workflow. Existing explicit MLX configurations and successful manifests remain usable. This document describes only the optional older Ollama profile. Requirement extraction, summaries, decision analysis, workflows and PRD generation are outside this toolkit.

The hybrid path is:

```text
Original media (read only)
  → FFmpeg normalization/extraction → optional local Silero VAD
  → one processed WAV + original-time mapping
  → Saaras V4 Batch, mode=codemix → retained original provider JSON
  → original-time Tamil–English transcript
  → local Ollama segment translation → English with the same IDs/timestamps
  → existing TXT/SRT/JSON writers and manifests
```

## Setup

Use the existing Python 3.12 environment and FFmpeg installed by `./setup.sh`. The Sarvam HTTP adapter uses the Python standard library and the official Batch endpoints, so no additional SDK is needed. Install Silero only when you select it:

```bash
uv pip install --python .venv/bin/python -r requirements-hybrid.txt
```

Silero uses the CPU and the model bundled in its installed package, rather than downloading through `torch.hub`. Its package dependencies include Torch/audio libraries and can be large. `vad_backend = "none"` needs none of these extra packages.

Install Ollama, start its local service, then explicitly download the model:

```bash
ollama serve
# In another Terminal, if it is not installed already:
ollama pull gemma3:12b
ollama list
```

Gemma 3 12B is the initial default for the Ollama backend. Allow sufficient RAM and disk space; inference speed depends on hardware, quantization, context size and concurrent workloads. The toolkit never downloads a local model automatically. Its default endpoint is `http://127.0.0.1:11434`; configurable endpoints must remain loopback origins. Cloud models/endpoints are rejected: preflight checks local tags for `remote_model`/`remote_host` metadata and cloud model tags as well as the endpoint. Keep Ollama cloud features disabled if you require a server-wide local-only policy.

Set `SARVAM_API_KEY` in your shell using your usual local secret-management method. Do not put it in TOML, transcript files, command-line arguments, screenshots or issue reports. The toolkit reads it only from the environment, sends it only to the official HTTPS API, and does not write it to the manifest or diagnostics. Signed storage URLs are used in memory and are not retained in logs or manifests.

## Configure and preview

Start with `config.hybrid.toml`, which leaves uploads unauthorized and sets a zero budget until you choose a ceiling. Paths are relative to the TOML file. Existing precedence still applies: CLI > TOML > built-in defaults.

```bash
mozhiflow --config config.hybrid.toml --dry-run "saaras-test/sample.wav"
```

Hybrid dry-run performs **local** audio extraction/normalization and optional VAD in a temporary directory; it makes no provider requests, uploads nothing, and creates no output directory. It prints original/audio/retained/removed durations, predicted duration savings, a deterministic mapping, and estimated recognition cost. Legacy MLX dry-run remains a configuration-only plan unless preprocessing is explicitly enabled.

Relevant settings beyond the sample configuration:

| Setting | Purpose |
|---|---|
| `recognition.backend` | `mlx_whisper` (legacy default) or `sarvam` |
| `recognition.model` | MLX repository/path or exactly `saaras:v4` for Sarvam |
| `recognition.language` | `ta`/`ta-IN`, other supported codes, or `auto`/`unknown` for Saaras |
| `recognition.vocabulary` | Saaras V4 keyterms; at most 50 distinct strings of at most 64 characters |
| `recognition.prompt` | MLX context only; rejected for Saaras to prevent silently ignored configuration |
| `preprocessing.enabled` | Opt-in preprocessing for MLX; Saaras always extracts a WAV |
| `preprocessing.normalize` | FFmpeg loudness normalization; mono 16 kHz PCM extraction always applies to Saaras |
| `preprocessing.vad_backend` | `none` or `silero`; Silero requires `enabled=true` |
| `preprocessing.threshold` | Speech probability threshold, default 0.35 |
| `preprocessing.min_silence_seconds` | Minimum removable gap interior after padding, default 5 seconds |
| `preprocessing.padding_seconds` | Preserve both speech boundaries, default 0.75 seconds |
| `preprocessing.min_speech_ms` | Silero minimum detected speech duration, default 250 ms |
| `preprocessing.max_intervals` | Reject excessive fragmentation, default 200 retained intervals |
| `sarvam.diarization` | Optional speaker-attributed output; different rate, default false |
| `sarvam.budget_inr` | Required positive cumulative estimate ceiling for new jobs under this output root |
| `sarvam.rate_per_minute` | Configurable non-diarized estimate rate |
| `sarvam.diarization_rate_per_minute` | Configurable diarized estimate rate |
| `sarvam.cost_margin` | Estimate multiplier, minimum 1; default 1.2 |
| `sarvam.poll_interval` / `poll_timeout` | Poll 0.1–30 seconds apart; stop waiting safely after the timeout |
| `translation.backend` | `mlx_whisper` (translate audio) or `ollama` (translate the transcript) |
| `translation.model` | Ollama defaults to `gemma3:12b`; MLX translation still requires Large V3 |
| `translation.prompt` | Extra translation guidance; changing it affects only translation |
| `translation.endpoint` | Local Ollama HTTP(S) origin |
| `translation.group_segments` | Maximum target segments per request, default 8 |
| `translation.context_segments` | Maximum preceding/following neighbors, default 1 on each side |
| `translation.max_chars` | Bound on target plus neighbor text, default 4000 characters |
| `translation.retries` | Retries per failed translation group, default 2, maximum 5 |
| `translation.options` | Ollama temperature, seed, num_ctx, num_predict, top_p/top_k/repeat_penalty |
| `runtime.request_timeout` | Per-request network timeout |
| `runtime.offline` | Prevent new Sarvam network calls when true; archived results and local Ollama remain available |

CLI overrides include `--recognition-backend`, `--model`, `--translation-backend`, `--translation-model`, `--translation-prompt`, `--ollama-endpoint`, `--sarvam-budget`, and the existing language/vocabulary/paths/output switches. Use TOML for the remaining knobs. Unknown keys and incompatible options fail validation.

## Explicit upload permission and cost guardrails

After reviewing the dry-run and choosing a budget, the following **explicitly authorizes uploading the selected recording**:

```bash
mozhiflow --config config.hybrid.toml --sarvam-budget 1 --allow-upload "saaras-test/sample.wav"
```

`--allow-upload` is a CLI-only consent flag: putting permission in TOML is rejected. A batch invocation authorizes its selected inputs, subject to the shared cost reservation ceiling. Consent is recorded against the exact processed audio hash and job; continuation of that job can reuse that grant. A new source/recognition configuration requires new consent. Translation-only processing never uploads audio.

As checked on 2026-10-18, the official [pricing page](https://www.sarvam.ai/api-pricing) lists pay-as-you-go Batch recognition at **₹0.49/minute**, and Batch with diarization at **₹0.59/minute**. These are configurable estimates, not contractual pricing. This toolkit rounds each file up to a whole minute and applies the configured margin. The 25-second sample estimates ₹0.588 without diarization at the default 1.2 margin. VAD savings estimate duration/cost reduction, not processing latency or recognition quality.

Reservations are stored under `.sarvam-budget.json` with a shared file lock, so concurrent recordings cannot exceed the configured **local estimate** ceiling. Reservations persist across invocations and are not automatically refunded after timeouts, ambiguous requests or failures. Increasing a ceiling must be deliberate. Estimates exclude taxes and cannot enforce your provider account's actual balance, provider rounding changes, or altered rates. Provider quota/authentication errors fail without creating replacement jobs.

Before a **new** paid job, if Ollama translation is enabled, the toolkit checks that its configured local model is installed. Once a remote job already exists, it can still be polled/downloaded independently so missing local translation availability does not discard a paid transcript. Credentials, consent, budget, duration limits and local prerequisites are checked before audio upload. Batch is used for short samples too, providing one durable workflow; recordings over the documented 2-hour per-file limit are rejected rather than silently split or truncated.

## VAD and timestamp semantics

FFmpeg produces mono 16 kHz signed-16-bit WAV audio. VAD changes only which portions of the extracted audio are retained; the original recording is never opened for writing. Normalization is optional. Silence removal preserves padding and leaves short gaps intact, so the toolkit does not turn every utterance into a separate upload. No-speech detection falls back to retaining the full audio, with metadata, rather than silently deleting it. Conservative settings reduce but cannot eliminate speech-loss risk: quiet speech, overlapping speakers, noise and accent can fool VAD. Preview and listen to important material; set `vad_backend="none"` when uncertain.

The mapping records integer original/processed sample intervals and the audio stream's offset relative to the recording timeline. At a removed-gap boundary, chunk starts map to the next retained interval and ends map to the preceding one. If a provider chunk crosses a removed gap, its original outer interval spans that gap, and `original_spans` lists the actual retained pieces. The toolkit does not split the text or invent sub-phrase alignment. Ollama copies the source's IDs, start/end and original_spans exactly. Saaras provides chunk-level, not word-level, timing.

The manifest records preprocessing configuration, source identity, FFmpeg/Silero versions, retained/removed durations, mapping and prepared audio hash. Preprocessing caches are versioned by configuration. After changing installed VAD/model implementations, choose a new output root for a fresh preparation; existing versioned metadata remains available.

## Resumption and selective processing

```bash
# Resume the same pending remote job (no new upload grant needed for that saved job):
mozhiflow --config config.hybrid.toml "path/to/meeting.mov"

# Retry local translation only; requires an intact transcript checkpoint:
mozhiflow --config config.hybrid.toml --only translation "path/to/meeting.mov"

# After changing only the local model, prompt or generation settings:
mozhiflow --config config.hybrid.toml --only translation --overwrite "path/to/meeting.mov"
```

The existing source/stage checkpoints, output hashes and no-overwrite behavior remain in place. Output conflicts still require explicit overwrite or a new output directory. Translation settings do not enter the recognition fingerprint. Saaras additionally reuses raw responses across output-format/name changes, even when final artifacts must be regenerated; `--overwrite` does **not** mean "buy another transcription of identical audio". Ollama validates and checkpoints each group, permitting a failed later group to resume without regenerating completed groups. A changed model tag/prompt/options invalidates translation groups. Model digests are recorded on actual generation; changing weights behind the same tag requires explicitly reprocessing translation to discover that change.

Remote files live under `.providers/sarvam/<fingerprint>/`: `remote.json` contains job/phase information, and `provider-response.json` retains the exact downloaded transcription response. Create/start intentions are saved **before** mutation. Reads are retried; an uncertain create is not automatically repeated. If create returned no usable job ID after a timeout, reconcile the request in Sarvam's dashboard and bind the existing job with `--sarvam-job-id JOB_ID`. Only use the job corresponding to the exact source/prepared WAV and requested recognition parameters. If a start outcome is uncertain and the server still reports Accepted, inspect/start that same job through the provider rather than blindly starting a replacement. Pending-job polling timeouts are safe to rerun. Failed remote jobs are never silently replaced.

Do not delete provider state or the budget ledger as a retry technique: losing remote identity can cause duplicate billable work. An orphaned reservation intentionally favors avoiding accidental spend; reconcile it with the provider before any manual change. The provider's original response is retained even when response shape or timestamp mapping is invalid. Source/translation output retry does not re-upload that response's audio.

## Local translation validation and limitations

Requests use Ollama JSON-schema structured output with fixed segment IDs and bounded groups, including neighboring context only within the character bound. Oversized individual source segments fail clearly instead of silently truncating/splitting their timing. Context size is token-based while `max_chars` is a character limit; Tamil tokenization may require a larger `num_ctx` or smaller groups. Incomplete/truncated generation and invalid structured responses are retried and then reported as failures.

Validation checks IDs/order/count, nonempty text, exact written numbers, source acronyms, recognizable identifiers and present technical keyterms. Dates/numbers must retain their written form; a semantically valid date reformat can therefore trigger a retry. The model is instructed to preserve names, negation, conditions and uncertainty, but no mechanical check here proves those semantic properties or detects every hallucination. Review against source audio and text. Code-mixed recognition/keyterm prompts likewise do not guarantee faithful spelling or preservation of all repetitions. Saaras codemix and verbatim are distinct provider modes; this path deliberately selects codemix and does not rewrite it afterward.

Local raw generation responses and validated group results are retained under `.providers/ollama/`. The manifest records provider, model tag/digest, generation parameters, group progress, source transcript hash, remote job ID, durations, cost estimate and statuses. Ollama failures cannot call Saaras because the translation adapter only reads the existing transcript checkpoint.

## Short-sample validation and comparison

`validation/hybrid-sample.toml` targets the existing 25-second `saaras-test/sample.wav`, disables VAD removal, selects Saaras V4 codemix + Gemma 3 12B, and caps estimated spend at ₹1. It still requires credentials and **explicit upload approval**:

```bash
.venv/bin/python scripts/validate_hybrid.py --config validation/hybrid-sample.toml --allow-upload
```

The utility refuses long samples, verifies source identity, all six output files, timestamp/ID alignment and a translation-only skip, then writes a textual comparison against `saaras-test/translate.json`. Differences are review material, not an automatic translation accuracy score. It always leaves `translation_quality_verified=false`; a person must listen and assess faithfulness.

To use your own archived Saaras codemix response with **no upload**, copy `config.hybrid.toml` to `config.import.local.toml` and configure the response binding described below:

```bash
.venv/bin/python scripts/validate_hybrid.py --config config.import.local.toml
```

Import configuration sets `sarvam.response_file` and the exact source file's `sarvam.response_source_sha256`. It requires VAD disabled, as an archived original-audio response cannot be remapped using a different removed-silence timeline. Imported results are labeled as user-supplied, with their original provider model version unverified and zero new provider cost. They are not evidence of a live Batch integration run. The sample configuration is specific to the current local sample bytes; changing that sample requires updating its explicit binding.

Run all regression tests using:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

See `validation/MOZHIFLOW_RENAME_REPORT.md` for automated verification results and limitations. Historical private-sample reports remain local. Unit tests mock VAD decisions and provider/model inference; the FFmpeg extraction/copy paths also run on synthetic audio. None of those tests establishes Tamil/English recognition or translation quality.

References: [Saaras/keyterms](https://docs.sarvam.ai/api/getting-started/models/saaras), [Batch API](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/batch-api), [Batch initiation](https://docs.sarvam.ai/api-reference/speech-to-text/stt/job/initiate), [Ollama generation/structured format](https://docs.ollama.com/api/generate), [Silero VAD](https://github.com/snakers4/silero-vad).
