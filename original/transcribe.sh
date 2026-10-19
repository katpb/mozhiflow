#!/usr/bin/env bash
# Transcribe one media file or a directory of media files with MLX Whisper.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="$ROOT/.venv/bin/mlx_whisper"
MODEL='mlx-community/whisper-large-v3-mlx'
LANGUAGE='ta'
OUTPUT_ROOT="$HOME/Transcripts"
PROMPT=''
FORCE=0
DRY_RUN=0
INPUT=''

usage() {
  cat <<'HELP'
Usage:
  ./transcribe.sh [options] "/path/to/recording.mov"
  ./transcribe.sh [options] "/path/to/folder"

Options:
  --language CODE     Default: ta (Tamil). Use 'auto' for auto detection, 'en' for English.
  --fast              Use multilingual large-v3-turbo (faster, may be less accurate).
  --model REPO         Override the Hugging Face MLX Whisper model repo/path.
  --output-dir DIR     Output root (default: ~/Transcripts).
  --prompt TEXT        Vocabulary hint, e.g. names or specialist jargon.
  --force              Re-transcribe even if output already exists.
  --dry-run            Show commands without running transcription.
  --help               Show this help.

Files accepted: .mov, .mp4, .m4a, .mp3, .wav, .aac, .flac, .mkv, .webm
Outputs: .txt, .srt, .vtt, .tsv, .json; one directory per source basename.
Notes: Files are processed locally after the model's first download.
HELP
}

while (( $# )); do
  case "$1" in
    --language|--model|--output-dir|--prompt)
      flag="$1"
      if (( $# < 2 )); then echo "Missing value for $flag" >&2; exit 2; fi
      case "$flag" in
        --language) LANGUAGE="$2" ;;
        --model) MODEL="$2" ;;
        --output-dir) OUTPUT_ROOT="$2" ;;
        --prompt) PROMPT="$2" ;;
      esac
      shift 2 ;;
    --fast) MODEL='mlx-community/whisper-large-v3-turbo'; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) if [[ -n "$INPUT" ]]; then echo 'Provide exactly one input file or folder.' >&2; exit 2; fi
       INPUT="$1"; shift ;;
  esac
done

if [[ -z "$INPUT" ]]; then usage >&2; exit 2; fi
if [[ ! -e "$INPUT" ]]; then echo "Input does not exist: $INPUT" >&2; exit 1; fi
if (( DRY_RUN == 0 )) && [[ ! -x "$CLI" ]]; then
  echo "MLX Whisper is not installed. Run: $ROOT/setup.sh" >&2
  exit 1
fi

is_supported() {
  local f
  f="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$f" in
    *.mov|*.mp4|*.m4a|*.mp3|*.wav|*.aac|*.flac|*.mkv|*.webm) return 0 ;;
    *) return 1 ;;
  esac
}

SUCCESS=0
FAILED=0
SKIPPED=0
process_one() {
  local media="$1" relative="$2"
  if ! is_supported "$media"; then return 0; fi
  local stem name out_dir
  name="${relative##*/}"
  stem="${name%.*}"
  # Mirroring relative subdirectories prevents same-named recordings in different directories colliding.
  if [[ "$relative" == */* ]]; then
    out_dir="$OUTPUT_ROOT/${relative%/*}/$stem"
  else
    out_dir="$OUTPUT_ROOT/$stem"
  fi
  if [[ -s "$out_dir/$stem.txt" && "$FORCE" == 0 ]]; then
    printf 'SKIP existing: %s\n' "$out_dir/$stem.txt"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  local args=("$CLI" "$media" --model "$MODEL" --task transcribe
    --output-format all --output-dir "$out_dir" --output-name "$stem" --verbose False)
  if [[ "$LANGUAGE" != 'auto' ]]; then args+=(--language "$LANGUAGE"); fi
  if [[ -n "$PROMPT" ]]; then args+=(--initial-prompt "$PROMPT"); fi
  printf 'TRANSCRIBE: %s\n' "$media"
  printf '       TO: %s\n' "$out_dir"
  if (( DRY_RUN == 1 )); then
    printf '       COMMAND: '; printf '%q ' "${args[@]}"; printf '\n'
    SUCCESS=$((SUCCESS + 1))
    return 0
  fi

  mkdir -p "$out_dir"
  # The upstream CLI can swallow per-file exceptions, so validate an actual nonempty output too.
  if "${args[@]}" && [[ -s "$out_dir/$stem.txt" ]]; then
    printf 'DONE: %s\n' "$out_dir/$stem.txt"
    SUCCESS=$((SUCCESS + 1))
  else
    printf 'FAILED: %s (no nonempty transcript generated)\n' "$media" >&2
    FAILED=$((FAILED + 1))
  fi
}

if [[ -f "$INPUT" ]]; then
  process_one "$INPUT" "${INPUT##*/}"
elif [[ -d "$INPUT" ]]; then
  # Recursive batch: preserves directory hierarchy inside ~/Transcripts.
  while IFS= read -r -d '' media; do
    rel="${media#"${INPUT%/}/"}"
    process_one "$media" "$rel"
  done < <(find "${INPUT%/}" -type f -print0)
else
  echo "Input is neither a regular file nor a directory: $INPUT" >&2
  exit 1
fi
printf '\nSummary: %d completed, %d skipped, %d failed\n' "$SUCCESS" "$SKIPPED" "$FAILED"
if (( FAILED > 0 )); then exit 1; fi
