# MozhiFlow rename and migration

MozhiFlow is the display name. The repository, distribution, Python package, and primary command are `mozhiflow`. Its description is: "Multilingual audio and video transcription, translation, and speaker-aware transcripts."

## Compatibility

Run `./setup.sh --sarvam-only`, activate `.venv`, then use `mozhiflow`. Use `./setup.sh` to include MLX Whisper. `python -m mozhiflow`, `./transcribe.sh`, and `python pipeline.py` accept the same processing options. The six former top-level Python modules remain aliases to their `mozhiflow` implementations, including when installed from a wheel. Existing monkeypatches and exception identities are preserved.

Source/editable installations keep the default `config.toml` beside the checkout. Wheel installations look in the current directory. Explicit `--config` works everywhere; paths inside TOML remain relative to that TOML. CLI paths remain relative to the working directory.

No stage names, configuration keys, fingerprints, manifests, provider job tokens, reservations, consent records, speaker namespaces, output templates, or model identifiers were renamed. Sarvam Batch recovery, MLX's isolated worker, Ollama, timestamps, diarization, preprocessing, budget controls, and 1Password injection retain their processing behavior. Tamil defaults remain configurable; provider language support is unchanged.

## Reference audit

The original project name appeared in the root README, historical `original/README.md`, and the private `config.toml` and `config.diarized.toml`. Current onboarding and historical installation paths now use `~/tools/mozhiflow`. Portable configuration examples already used relative paths.

The two private configurations still name the existing directory deliberately. Absolute old paths also occur in private validation evidence, manifests, preprocessing records, inference results, virtual-environment activation scripts, and executable shebangs. These are operational state or historical evidence, not portable installation defaults. Do not replace strings across those files. Source identity includes the resolved absolute path; moving recordings can invalidate checkpoints even when their bytes are unchanged.

There was no existing Python package metadata. The new package uses relative internal imports and exposes the console command through `pyproject.toml`. Old top-level files are compatibility modules. Public validation reports keep the commands that were actually run historically. `AGENTS.md` is preserved.

## Local directory migration

The working directory has **not** been relocated. Process enumeration is restricted in this execution environment. At inspection, all 13 discovered pipeline/budget locks were unlocked, and saved remote records were downloaded or imported; this does not establish that every external process is idle.

Before changing any paths:

1. Stop new submissions. Check Terminal, background workers, local checkpoint locks, and the Sarvam dashboard for unfinished or ambiguous jobs. Reconcile existing IDs; do not submit replacements.
2. Back up recordings, complete output trees (including hidden files), configurations, and secrets through your usual private backup mechanism. Keep provider state and `.sarvam-budget*` ledgers together with their outputs.
3. Prefer keeping existing recordings and recoverable job trees at their original canonical paths. A new source checkout at `~/tools/mozhiflow` can use an explicit private TOML pointing to those existing input/output paths.

For immediate access through the new directory name while retaining canonical checkpoint paths, create an alias manually from outside the project, only if the destination does not already exist:

```bash
cd ~/tools
test ! -e mozhiflow && test ! -L mozhiflow && ln -s tamil-whisper-local mozhiflow
cd ~/tools/mozhiflow
source .venv/bin/activate
mozhiflow --help
```

This is a directory alias, not a physical move. A physical `mv` of the whole project requires a separate state migration review: resolved source paths and retained audio paths may change, so a simple directory rename is insufficient for reliable paid-job recovery. Do not edit manifests or delete reservations to make retries work. Keep the original tree until recovery is verified.

Virtual environments are not relocatable: recreate `.venv` and rerun setup after relocating source. Editable installation metadata and generated command shebangs must point at the new checkout. Do not copy the old `.venv` into a fresh installation. Verify an explicit-config dry run before processing anything.

## Repository and GitHub

This directory contains no `.git` metadata, so history, branches, remotes, and the GitHub owner could not be inspected. No repository was initialized, remote renamed, or changes pushed.

In the actual Git checkout, retain `.git` and review these changes normally. With explicit approval, rename the GitHub repository to `mozhiflow` in its settings, then update the existing checkout's remote, substituting its actual owner:

```bash
git remote -v
git remote set-url origin git@github.com:OWNER/mozhiflow.git
git remote -v
```

Review any external CI, shortcuts, documentation links, and secret-manager launch commands that refer to the old repository or directory. Keep each developer's `.env.1password` private; the new shareable example contains placeholders only.
