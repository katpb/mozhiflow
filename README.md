# MozhiFlow

Multilingual audio and video transcription, translation, and speaker-aware transcripts.

This toolkit processes local MOV, MP4 and common audio recordings through the `mozhiflow` CLI. Its primary workflow is:

```text
Video → local FFmpeg extraction ─┐
Audio → audio preprocessing ─────┤
                                ↓
                 Optional conservative Silero VAD
                                ↓
                  ONE consolidated FLAC (or WAV)
                        ↙                 ↘
              Saaras V4 Batch       Saaras V4 Batch
                codemix                 translate
                        ↘                 ↙
                   TXT / SRT / JSON + manifest
```

An eligible recording uses **one job per enabled mode**, normally two jobs total. Removing silence does not create jobs per speech interval. Both jobs read the same prepared file and map their own provider timestamps to the original recording. They do not translate one another's text or force their segment boundaries to align. Originals are only read and are never removed by the toolkit.

MLX Whisper remains supported through existing configurations. The older optional Saaras/Ollama profile remains available in [HYBRID.md](HYBRID.md); Ollama is **not required** for the primary workflow. Summaries, requirement extraction and PRD generation are outside this project.

## macOS setup

Requires Homebrew, isolated Python **3.12**, FFmpeg/FFprobe, an internet connection for Sarvam, and a Sarvam account with credits. Sarvam-only operation does not require Apple GPU access or local model weights.

Clone repository `mozhiflow` or place its source in `~/tools/mozhiflow`. The examples below use that directory; any permanent checkout location works:

```bash
mkdir -p ~/tools
git clone https://github.com/katpb/mozhiflow.git ~/tools/mozhiflow
cd ~/tools/mozhiflow
# Install isolated Python 3.12, FFmpeg and uv if absent; no MLX model/dependency required:
./setup.sh --sarvam-only
# Portable settings for another developer:
cp config.example.toml config.toml
mkdir -p input
source .venv/bin/activate
mozhiflow --version
mozhiflow --help
```

`setup.sh` uses Homebrew to install FFmpeg and uv if needed. You can install them manually with `brew install ffmpeg uv`. Verify `ffmpeg -version`, `ffprobe -version`, and `.venv/bin/python --version`.

Allow working disk space for retained intermediates: a two-hour mono 16 kHz/16-bit PCM WAV is about 230 MB, and extraction/consolidation may temporarily keep several copies before FLAC compression.

For the legacy MLX backend on Apple Silicon, run `./setup.sh` without `--sarvam-only`. Existing environments are reused. Python, packages and models are not installed into another project's environment. The CLI requires Python 3.12 and uses the standard-library `tomllib` parser.

## Package and command

The distribution, Python package, repository, and CLI are named `mozhiflow`; the display name is **MozhiFlow**. Setup installs it in editable mode. Activate `.venv` in each new shell, or use `.venv/bin/mozhiflow` directly. `python -m mozhiflow` runs the same command. `./transcribe.sh` and `python pipeline.py` remain compatibility entry points with the same options and checkpoint behavior.

For a manual installation with Python 3.12 or later and FFmpeg/FFprobe already installed:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
# Optional Apple Silicon local recognition:
uv pip install --python .venv/bin/python -e '.[mlx]'
# Optional local silence detection:
uv pip install --python .venv/bin/python -e '.[vad]'
source .venv/bin/activate
cp config.example.toml config.toml
mkdir -p input
```

The Sarvam and Ollama adapters use the Python standard library; Ollama itself and its model are installed separately. Configurations retain the existing Tamil defaults; select a provider-supported language or `auto` for other languages. This rename does not change provider language capabilities.

Existing users should read [the rename and migration guide](docs/RENAMING.md) before relocating a checkout. Keep the original paths and state for ongoing or recoverable paid jobs.

## Configure your own Sarvam key

Create a key and check credits in the [Sarvam dashboard](https://dashboard.sarvam.ai/). The toolkit reads the **value** from `SARVAM_API_KEY` in the process environment. TOML stores only the environment-variable name. Never put a key value in configuration, source code, command-line arguments, logs or Git.

### 1Password

Install the 1Password CLI and enable its usual desktop/account integration. Each developer supplies their own vault/item references:

```bash
cp .env.1password.example .env.1password
chmod 600 .env.1password
```

Edit the private `.env.1password` to reference your own field, for example:

```dotenv
SARVAM_API_KEY="op://YOUR_VAULT/Sarvam/api_key"
```

This is a **secret reference**, not a key value. `.env.1password` is ignored by Git; the shareable example contains placeholders only. Existing secrets/references are never read, changed or rotated by setup. Runtime injection uses the standard command:

```bash
op run --env-file=.env.1password -- mozhiflow --config config.toml
```

For a fresh recording, use the explicit upload flag described below. For non-1Password users, supply the same environment variable through your preferred secret manager or a private shell session, then run `mozhiflow` directly. Do not paste the key into a command saved to shell history. A private environment file is **not** automatically sourced by the shell wrapper; use `op run` or an existing securely injected environment.

## Configuration and paths

`config.toml` defaults to the source checkout directory for editable/source runs and the current working directory for an installed wheel. Use `--config /path/to/config.toml` to select it explicitly. It is machine-specific and ignored by Git. `config.example.toml` is the portable, complete working example.

Use portable paths in new configurations:

```toml
[paths]
input = "input"
output_dir = "Transcripts"
```

The portable example uses `input` and `Transcripts` relative to itself. Relative TOML paths resolve from the TOML file; CLI paths resolve from the current directory. `~` expands, but `$VARIABLE` inside TOML does not. Configuration precedence is **CLI > TOML > built-in defaults**. An explicitly named missing config, unknown key, unsupported backend/mode/format, invalid type, unsafe filename, or incompatible codec/rate fails clearly.

| Section | Settings and behavior |
|---|---|
| `paths` | `input` is one file or a directory; `output_dir` is the result root |
| `preprocessing` | `enabled`, `codec="flac"` or `"wav"`, `sample_rate`, `normalize`, `retain_intermediates` |
| `preprocessing` VAD | `vad_backend="none"` or `"silero"`, `threshold`, `min_silence_seconds`, `padding_seconds`, `min_speech_ms`, `max_intervals` |
| `recognition` | `backend="sarvam"`, `model="saaras:v4"`, `mode="codemix"`, `language="ta-IN"` or `ta`/`auto`, `vocabulary` |
| `translation` | `backend="sarvam"`, `model="saaras:v4"`, `mode="translate"` |
| `sarvam` authentication | `api_key_env="SARVAM_API_KEY"` names the environment variable; it never contains the key |
| `sarvam` timing | `with_timestamps=true`, `timestamp_policy="flag"` or legacy `"strict"` |
| `sarvam` cost | `budget_inr`, `budget_scope`, `rate_per_minute`, `diarization_rate_per_minute`, `cost_margin`, optional `diarization` / `with_diarization`, `num_speakers`, mode-specific `speaker_names` |
| `sarvam` Batch | `read_retries`, `poll_interval`, `poll_timeout`; one consolidated file per mode, no automatic splitting |
| `outputs` | Independent `transcript` / `translation` booleans, `formats`, `filename`, `directory` |
| `batch` | `recursive`, `continue_on_error` |
| `runtime` | `offline`, `request_timeout`, `overwrite` |

CLI overrides include input path, `--output-dir`, `--recognition-backend`, `--model`, `--language`, `--vocabulary`, `--translation-backend`, `--translation-model`, `--audio-codec`, `--audio-sample-rate`, `--retain-intermediates` / `--no-retain-intermediates`, `--filename`, `--output-directory`, `--sarvam-budget`, `--sarvam-budget-scope`, `--sarvam-key-env`, `--sarvam-timestamp-policy`, output toggles/formats, recursion, overwrite and offline mode. See `mozhiflow --help`. Use TOML for the other settings. For list options such as `--vocabulary` or `--formats`, put `--` before a following input path to prevent it being consumed as another list value.

Saaras V4 keyterms come from `recognition.vocabulary`: at most 50 distinct strings, each at most 64 characters. They bias recognition; they cannot guarantee spelling or preserve every English term. MLX's `recognition.prompt` is retained for MLX configurations and rejected for Saaras rather than silently ignored. Saaras translation does not accept local-LLM generation prompts.

## Preview without API charges

Place recordings in `input/`, or specify one directly:

```bash
mozhiflow --config config.toml --dry-run "input/தமிழ் meeting (1).mov"
# Preview the configured directory:
mozhiflow --config config.toml --dry-run
```

Dry-run inspects media locally and uses temporary files only when processing is needed. It prints original/audio/trimmed/removed durations, mappings, actual codec/rate/size, passthrough decisions with reasons, and configured job counts. The separate `budget_preflight` report accounts for saved jobs: configured cap/scope, previous committed estimates, outstanding reservations, each pending mode, remaining allowance, and whether new processing is permitted. Recovery of existing jobs requires no additional reservation. No Sarvam request, key, consent, ledger change or result directory is needed. A blocked preview exits with status 2. Configured job counts describe the requested workflow; inspect `recoverable` and `additional_reservation_inr` for new spending. Long recordings can take time to inspect/decode/VAD.

## Process one video or audio file

After reviewing the preview and budget, explicitly authorize the first cloud upload:

```bash
op run --env-file=.env.1password -- \
  mozhiflow --config config.toml --allow-upload "input/meeting.mov"
```

Without 1Password, run the same `mozhiflow` command in the securely configured environment. Filenames with spaces, parentheses and Unicode must be quoted. Accepted input extensions include MOV, MP4, MKV, WebM, WAV, FLAC, M4A, MP3, AAC, OGG, OPUS, AIFF and WMA.

Audio selection uses ffprobe streams and container information, not the filename alone. A full packet scan rejects demux errors/empty streams; it does not certify every compressed frame through decoding. A supported single-stream audio source is **uploaded unchanged** when normalization is off and VAD is `none`, even with `preprocessing.enabled=true`. Stereo AAC in M4A, AAC/ADTS, MP3, FLAC, PCM WAV, Ogg Opus/Vorbis, and PCM AIFF are eligible at their native sample rate. Cover art is not treated as video. Unknown codec/container pairs or multiple audio streams fall back to conversion of the first stream. The upload extension and MIME follow the actual format, including when a file has a misleading suffix. See [Sarvam supported formats](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/overview).

Video still follows FFmpeg extraction. Enabled normalization or VAD requires one consolidated processed file shared by both modes; no separate request is made per speech interval. Configured codec/sample rate describe converted output, not a requirement to transcode otherwise compatible audio. The manifest and preview record `audio_extraction_skipped`, `audio_conversion_skipped`, `source_audio_reused`, and a reason for each decision. Passthrough uses identity timestamp mapping at the actual sample rate and audio timeline offset. FLAC conversion is lossless relative to extracted PCM, but resampling/downmixing changes the audio. Normalization defaults off. Original recordings are never overwritten or removed.

An upgrade retains a previously prepared payload if its paid or ambiguous remote job already exists. This preserves job tokens and avoids buying replacement work merely because native passthrough is now available.

Prepared audio is retained for both jobs and later retries. `retain_intermediates=false` removes the normalized scratch WAV after successful preparation, while keeping the consolidated upload payload, mapping, provider responses and checkpoints needed for safe resumption. Directly reused source audio is never deleted by any retention setting.

## Process multiple recordings

```bash
# First authorize the selected directory's uploads, subject to the shared budget:
op run --env-file=.env.1password -- mozhiflow --config config.toml --allow-upload
# Resume or skip the already authorized work:
op run --env-file=.env.1password -- mozhiflow --config config.toml
```

Recursion mirrors relative input subdirectories. The default new layout uses each source stem as its directory. If `meeting.mov` and `meeting.mp4` would share one destination, the whole discovery phase rejects that collision **before processing or charging**; use `outputs.directory="{name}"` for the old extension-qualified layout. The complete batch cost is checked before any new paid submission. A blocked preflight stops the invocation with one clear error; successful prior work is retained.

Upload authorization is separate from credentials and network enablement. `--allow-upload` is CLI-only; granting it through TOML is rejected. The manifest records consent for selected modes and the exact prepared-audio hash plus recognition parameters. This lets a later invocation resume the same authorized two-mode workflow after an interruption, even if the second job had not yet been created. A new recording, unapproved additional mode, or materially changed recognition parameters requires fresh consent.

## Optional conservative silence removal

Silence removal is **off** with `preprocessing.vad_backend="none"`. To enable local Silero:

```bash
uv pip install --python .venv/bin/python -r requirements-hybrid.txt
```

Then set:

```toml
[preprocessing]
enabled = true
vad_backend = "silero"
sample_rate = 16000
threshold = 0.35
min_silence_seconds = 5.0
padding_seconds = 0.75
min_speech_ms = 250
max_intervals = 200
```

Silero uses bundled local model weights and CPU inference. Padding is applied to both speech boundaries, and only remaining gap interiors of at least the configured silence duration are removed. Short pauses stay intact. No detected speech conservatively retains the whole recording rather than silently deleting it. Too many retained intervals fail as excessive fragmentation. Silero accepts 8/16 kHz here; without VAD the extractor also supports 24/48 kHz.

Retained intervals are combined into **one** WAV/FLAC. Even with hundreds of intervals, the normal paid workflow remains one codemix job and one translate job. Quiet speech, noise, accents and overlap can cause detection errors; preview and review boundaries or disable VAD when uncertain. No real Silero recall/quality claim is made from mocked tests.

## Outputs and timestamp interpretation

```text
Transcripts/meeting/
  meeting.codemix.txt
  meeting.codemix.srt
  meeting.codemix.json
  meeting.translate.txt
  meeting.translate.srt
  meeting.translate.json
  manifest.json
  .preprocessing/…
  .providers/sarvam/…
  .inference/…
```

`outputs.filename="{stem}.{mode}"` produces the names above. Existing `{stem}.{stage}` templates continue to produce `transcript`/`translation` suffixes. Templates also support `{extension}`; directory templates support `{stem}`, `{name}` and `{extension}`. Path traversal and formatting expressions are rejected.

TXT retains full provider text without cleanup, rewriting or summarization. JSON holds normalized original-time segments, processed timestamps, actual retained `original_spans`, provider/model/mode and request metadata. Exact provider bytes remain separately under `.providers/sarvam/<fingerprint>/provider-response.json`. Translation JSON identifies English output separately from the provider's detected source-language code.

The manifest contains source identity/SHA-256, effective configuration, original and prepared durations, codec/rate/bytes, removed intervals, sample-based mapping, preprocessing versions, upload authorization, per-stage provider/model, remote job IDs/statuses, uploaded duration, estimated cost, attempts, checkpoints, output hashes and completion/failure status. Imported archived responses are labeled as imports with zero uploaded duration/new cost and unverified original provider model version.

Each mode independently maps its own chunk timestamps to the original recording. At a removed-gap boundary, starts map to the next retained interval and ends to the preceding interval. A chunk crossing removed silence keeps its original outer bounds, with the actual retained pieces listed as `original_spans`; no word/phrase precision is invented. Translated and codemix chunks may cover different intervals. Their IDs are local to each output and do not imply cross-mode alignment.

If the provider omits timestamps or supplies positions that cannot be mapped reliably, `timestamp_policy="flag"` retains text and records the problem with `start/end=null`, `mapping_error`, and `_timing` metadata. Invalid/untimed cues are omitted from SRT with a warning. If all speech is untimed, SRT is empty and explicitly incomplete: inspect TXT/JSON and the raw response. `strict` retains the original provider response but fails that stage instead. Non-finite numeric values are encoded as JSON null with their original labels/paths recorded; no fake intervals are generated. A completed stage is not a claim of linguistic or subtitle accuracy.

## Retry without duplicating charges

```bash
# Resume both pending modes or skip intact results:
op run --env-file=.env.1password -- mozhiflow --config config.toml
# Retry English only; no codemix checkpoint is required:
op run --env-file=.env.1password -- mozhiflow --config config.toml --only translation
# Retry codemix only:
op run --env-file=.env.1password -- mozhiflow --config config.toml --only transcript
```

Each mode has its own job identity, raw response and checkpoint. One mode can fail while the other succeeds. Existing successful outputs are skipped only when settings and hashes match. Changed or manually edited results need explicit `--overwrite` or another output root. Source identity changes are rejected even with overwrite: choose a new root for genuinely different media. Per-recording locks prevent competing local writers; a shared locked ledger reserves estimated costs across recordings.

Intent is saved **before** create/start requests. Timeouts during polling resume the saved job. Definitive authentication/credit/validation rejections can be retried after fixing the cause. Ambiguous create failures do not blindly resubmit: reconcile the provider dashboard, then supply `--sarvam-job-id JOB_ID` for codemix or `--sarvam-translation-job-id JOB_ID` for translation. Those arguments reconcile saved uncertain creates; they do not silently create new jobs on fresh state. Use the exact job corresponding to that source, payload and mode.

An uncertain start with a server state of Accepted requires inspecting/starting that same existing job in the dashboard; it is not replaced automatically. Failed remote jobs are not automatically bought again. Inspect the provider failure before deliberately authorizing a new submission in a fresh output location. Do not delete remote state, the budget ledger or raw responses as a retry technique: losing job identity can cause duplicate spend.

Raw responses are reused for identical paid-inference parameters even when regenerating output formats, names or timestamp-policy handling. `--overwrite` explicitly permits artifact replacement but does **not** purchase another identical Saaras inference when its raw response is available. Changing the codec, speech selection, keyterms or model can change paid work and requires preview/consent. To resume an older run, retain its original output root, directory/filename templates and preprocessing settings; do not move it into the new layout mid-job.

## Speaker diarization and timestamped output

Saaras V4 Batch supports `with_diarization=true` alongside `with_timestamps=true`, with optional `num_speakers` from 1 to 20; omission means automatic speaker-count detection. See the [official diarization guide](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/how-to/enable-speaker-diarization). Configure the existing application:

```toml
[sarvam]
with_diarization = true # Alias for the existing diarization key; replace diarization=false
with_timestamps = true
# num_speakers = 2 # Omit for automatic detection
# Optional, manually reviewed labels, configured independently for each mode:
speaker_names = { codemix = { "0" = "Kathir" }, translate = { "1" = "Kathir" } }
```

Do not set conflicting `diarization` and `with_diarization` values. A speaker count requires diarization, and this application's speaker-attributed outputs require timestamps. No CLI or 1Password launch change is needed: use the existing `--dry-run`, then `--allow-upload` after reviewing cost and consent.

The application parses `diarized_transcript.entries` into TXT/SRT/JSON for both modes. TXT contains original-recording timestamp ranges and speaker labels; SRT adds the same label to each valid cue. JSON retains the full provider transcript, each entry's unchanged text and original provider `speaker_id`, optional configured `speaker_name`, display label/status, processed timestamps, remapped original timestamps and original spans across removed silence. Raw API response bytes and their checksums remain available through the manifest. Invalid timestamps are flagged/omitted according to the existing timestamp policy, never guessed; TXT marks unavailable times explicitly.

Example TXT line:

```text
[00:00:10,100 --> 00:00:12,800] Kathir: API integration...
```

**Speaker IDs belong to one provider job.** Codemix speaker `0` and translate speaker `0` are not assumed to be the same person; JSON records separate mode/job namespaces. Without configured names, labels include the mode (for example `codemix speaker 0`). Missing/unusable IDs are `unknown speaker`; explicitly uncertain assignments stay uncertain and do not receive a configured name. Confidence is retained when provided; the toolkit does not infer real identities or invent confidence scores. Speaker labels are clustering results, not verified identities. Review the raw output and recording before assigning names. Mode-specific name mappings apply to all selected recordings, so use a configuration for one reviewed recording when assigning personal names across independently generated jobs.

Existing successful/submitted stages are protected: toggling diarization or changing `num_speakers` in the same output location stops at preflight, including with `--overwrite`. Restore the original settings to recover those jobs. To deliberately obtain a new paid diarized result, choose a separate output directory with the existing `--output-dir` option, preview the combined cost, and explicitly authorize uploads there. Source media, existing results, remote IDs and reservations are retained. There is no automatic upgrade of an existing non-diarized transcript and no automatic replacement of a failed diarized job.

Changing only speaker names is a local rendering change and does not change the paid job token. After manual review, the existing `--overwrite` option can regenerate outputs from retained raw responses with the updated names; no new provider call is needed when those responses are intact. By default, successful output files remain untouched. Segments from separate modes are never artificially aligned, and speaker identities are never inferred across modes. Diarization accuracy, overlap handling and translation fidelity require inspection of real provider output; mocked tests cannot verify them.

## Batch limits and cost

Official documentation checked on 2026-10-19:

- Batch: up to **2 hours per file**, up to 20 files per job; this toolkit sends one consolidated file per mode.
- V4 modes use the official `POST /speech-to-text/job/v1` lifecycle: create, request upload URL, storage upload, start, GET status, request download URL, download JSON.
- Compatible native audio formats and extracted FLAC/WAV are auto-detected. No unsupported FLAC codec-enum value is sent.
- Chunk-level timestamps are supported; word-level precision is not promised.
- Pay-as-you-go Batch recognition remains configured at **₹0.49/minute**. The diarization default is now **₹0.75/minute** (₹45/hour), following the [official API pricing docs](https://docs.sarvam.ai/api/getting-started/pricing). The [website pricing page](https://www.sarvam.ai/api-pricing) currently lists ₹0.59/minute for diarized Batch; these official sources disagree. We use the higher rate conservatively until clarified. Verify your account rate; `diarization_rate_per_minute` remains configurable. Each mode is a separate paid job, and estimates are not account charges.

The local estimator rounds each file up to a whole minute and multiplies by `cost_margin` (default 1.2). With both modes, 60 minutes estimates ₹70.56 and 120 minutes ₹141.12 without diarization, excluding tax. With diarization at ₹0.75/minute and margin 1.2, those combined estimates are ₹108 and ₹216 respectively. The whole selected plan must fit the existing budget scope before any new job is created; enabling diarization does not raise the cap or reset history. Each mode has a distinct job token; rerunning the same token does not add its cost again. These are conservative local estimates, not reported Sarvam charges or account credits. Tax, account plans, provider rounding and pricing changes can alter real charges; check account credits separately.

Configure an explicit cap and scope:

```toml
[sarvam]
budget_inr = 150.0
budget_scope = "persistent" # persistent | recording | invocation
```

| Scope | What consumes the cap |
| --- | --- |
| `persistent` (backward-compatible default) | All reservations under this output root, including previous recordings and completed/failed jobs. |
| `recording` | All tracked jobs for the source checksum, including previous attempts/modes for that recording. Each recording gets the configured cap. |
| `invocation` | All pending new work across the selected files/modes in this CLI invocation. Prior paid-job recovery adds no cost. A later invocation receives a fresh cap explicitly because this scope was selected; history remains stored. |

Scope changes are explicit spending-policy changes, not refunds or ledger resets. The example and existing machine configuration keep `persistent`. CLI overrides use `--sarvam-budget` and `--sarvam-budget-scope`. A narrowly selected `--only translation` still honors outstanding persistent/recording reservations. Preview before authorizing:

```bash
mozhiflow --config config.toml --dry-run
# Explicitly choose a per-recording policy for this preview:
mozhiflow --config config.toml --sarvam-budget-scope recording --dry-run
```

The `.sarvam-budget.json` ledger retains legacy `reservations` and adds source/mode/invocation metadata for new reservations. Started, downloaded, and remotely completed/failed jobs count as committed estimates. Unstarted, unmatched legacy, and ambiguous reservations remain outstanding; none are automatically refunded. A definitive create rejection does not discard its reservation. Reserved-only work must still fit the chosen scope when retried. Existing remote IDs and uncertain creates are recovered/reconciled rather than submitted again; a recovery-only selection remains permitted after lowering the cap. If a selection mixes recovery with unaffordable new jobs, use `--only` to recover the existing mode separately or explicitly revise the spending policy.

All pending modes/files are checked together before the first new paid job. Reservations are saved atomically under a ledger lock; per-stage checks reuse them. A blocked preflight exits with status 2 before any new submission, showing committed/reserved costs, per-mode estimates and the shortfall. Processing manifests retain the budget breakdown and media-selection diagnostics without replacing paid-job history. Do not delete the ledger or remote state to bypass a spending limit.

If prepared audio exceeds two hours, processing stops before upload. Dry-run and errors show the theoretical number of pieces/jobs, for example four total jobs for two modes if two pieces per mode would be needed. **Automatic splitting is disabled.** Explicitly decide whether to supply shorter recordings or revise conservative silence settings before granting any additional-job workflow; the toolkit never raises the job count on its own.

Sources: [Batch guide](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/batch-api), [model/formats](https://docs.sarvam.ai/api/getting-started/models/saaras), [API initiation](https://docs.sarvam.ai/api-reference/speech-to-text/stt/job/initiate), [start](https://docs.sarvam.ai/api-reference/speech-to-text/stt/job/start), [pricing](https://www.sarvam.ai/api-pricing).

## Troubleshooting and privacy

- **Missing key:** inject the configured environment variable with `op run`; a reference file alone does not set the environment. Do not print environment values to debug.
- **Upload not authorized:** preview and deliberately use `--allow-upload` for new inputs/modes. Credentials are not consent.
- **Budget/credits:** review both-mode totals and the existing ledger. HTTP authentication/quota errors retain stage state; no replacement is silently charged.
- **Pending timeout:** rerun the same configuration to poll the existing ID; do not change output roots or delete state.
- **Missing/unmappable timestamps:** inspect manifest timing warnings, JSON and original provider bytes. Empty SRT can mean unavailable timing, not missing retained text.
- **No input:** place files under the configured `input` directory or pass one quoted path. Existing files in `inputs/` are not moved automatically when switching to `input/`.
- **Silero unavailable:** install the optional requirements or set `vad_backend="none"`. Noise/quiet speech remains a recognition risk with or without VAD.
- **Ollama/Metal errors:** neither is required by the Saaras-only profile. Check that you selected `sarvam` for both stages. Legacy MLX still requires Apple Silicon/Metal.
- **1Password prompts:** use your own existing account/item references; setup never changes secrets. Consult the local `op run --help` for your installed CLI.

Sarvam mode uploads the selected prepared audio to Sarvam/provider storage. Uploads use explicit consent and HTTPS; subscription keys are not attached to signed storage PUTs, and signed URLs/keys are not logged or persisted. Raw provider replies, speech text, audio intermediates and manifests are sensitive local data: secure their directory. `.gitignore` excludes credentials, local config, input media, default result roots, generated mode outputs, provider state and caches. Add any custom result root before sharing. Nothing is published or pushed by the toolkit.

Provider recognition/translation can omit words or repetitions, transliterate English, change jargon/names/numbers, lose negation/conditions/uncertainty or infer content during noise. The toolkit does not silently repair those words. Review important passages against the recording. Passing unit tests does not verify translation quality.

## Legacy MLX migration

Existing MLX TOML files remain valid: built-in defaults still select MLX and the old `{name}` directory / `{stem}.{stage}` filename layout. The shipped primary configuration now explicitly selects Sarvam. Keep an older run's original TOML/output root to resume its checkpoints; no existing media or outputs are moved into the new layout automatically.

For a new MLX run, create an ignored local configuration from the example and change these settings:

```toml
[recognition]
backend = "mlx_whisper"
model = "mlx-community/whisper-large-v3-mlx"
language = "ta"

[translation]
backend = "mlx_whisper"
model = "mlx-community/whisper-large-v3-mlx"

[preprocessing]
enabled = false
vad_backend = "none"

[outputs]
filename = "{stem}.{stage}"
directory = "{name}"
```

Keep the other sections from the example, run `./setup.sh` to install MLX, and use `--no-offline` for its initial model download. `--fast`, `--force`, `--prompt`, the existing output toggles and prior CLI stage names remain available. MLX still uses `task=transcribe` and multilingual Large V3 `task=translate`; the two audio passes are independent. The optional Ollama text-translation backend is also retained but is never invoked by the Sarvam-only profile.

## Developers and explicit integration validation

```bash
./setup.sh --sarvam-only
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q mozhiflow scripts tests
bash -n setup.sh transcribe.sh
```

Normal tests use fake provider responses and synthetic local FFmpeg fixtures. They never make real Saaras calls or upload private recordings. Optional Silero and MLX inference are mocked. The suite covers both modes, actual MOV extraction/FLAC/direct-audio reuse, consolidation/mapping, zero speech, boundaries, cost/security guards, independent failures, interruption, resume and output generation.

For an **explicitly requested** live integration test, copy the example to an ignored `config.local.toml`, point it at one consented sample of at most 30 seconds and a separate test output root, set `budget_inr=2.0`, then run:

```bash
op run --env-file=.env.1password -- \
  .venv/bin/python scripts/validate_hybrid.py --config config.local.toml --allow-upload
```

This is the existing short-sample validator, extended for the Sarvam-only profile; it calls the same CLI. It validates source/output integrity and a successful translation-only skip. A saved reference response can optionally be compared with `--reference PATH`; a textual diff is not a translation-quality score. The validator deliberately does not require identical chunk counts/times between independent Saaras modes and always leaves semantic quality unverified pending listening/review.

Packaging and rename verification are documented in [validation/MOZHIFLOW_RENAME_REPORT.md](validation/MOZHIFLOW_RENAME_REPORT.md). Provider validation evidence and remaining unverified behavior are documented in [validation/SAARAS_V4_REPORT.md](validation/SAARAS_V4_REPORT.md). No new live Batch submission or long-recording quality validation is claimed by the local replay and mocked tests.
