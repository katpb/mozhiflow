"""MozhiFlow: Multilingual audio and video transcription, translation, and speaker-aware transcripts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import string
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent.parent
# Source checkouts retain the legacy default; installed wheels use the working directory.
default_config_root = ROOT if (ROOT / 'pyproject.toml').is_file() else Path.cwd()
LARGE = 'mlx-community/whisper-large-v3-mlx'
TURBO = 'mlx-community/whisper-large-v3-turbo'
EXTENSIONS = {'.mov', '.mp4', '.m4a', '.mp3', '.wav', '.aac', '.flac', '.mkv', '.webm', '.ogg', '.opus', '.aiff', '.aif', '.wma'}
STAGES = ('transcript', 'translation')
WARNING = ('Translated subtitle timestamps may not exactly match source transcript segments. '
           'Each stage decodes the original audio independently; segments are not aligned.')


from .toolkit_core import PipelineError


@dataclass
class Config:
    input: str = 'input'
    output_dir: str = '~/Transcripts'
    model: str = LARGE
    language: str = 'ta'
    vocabulary: list[str] = field(default_factory=list)
    prompt: str = ''
    translation_backend: str = 'mlx_whisper'
    translation_model: str = LARGE
    transcript: bool = True
    translation: bool = True
    formats: list[str] = field(default_factory=lambda: ['txt', 'srt', 'json'])
    filename: str = '{stem}.{stage}'
    recursive: bool = True
    continue_on_error: bool = True
    overwrite: bool = False
    offline: bool = True
    recognition_backend: str = 'mlx_whisper'
    preprocess_enabled: bool = False
    normalize_audio: bool = True
    vad_backend: str = 'none'
    vad_threshold: float = 0.35
    vad_min_silence: float = 5.0
    vad_padding: float = 0.75
    vad_min_speech_ms: int = 250
    vad_max_intervals: int = 200
    sarvam_diarization: bool = False
    sarvam_num_speakers: int | None = None
    sarvam_speaker_names: dict = field(default_factory=dict)
    sarvam_budget: float = 0.0
    sarvam_budget_scope: str = 'persistent'
    sarvam_rate: float = 0.49
    sarvam_diarization_rate: float = 0.75
    sarvam_cost_margin: float = 1.2
    sarvam_poll_interval: float = 5.0
    sarvam_poll_timeout: float = 600.0
    sarvam_job_id: str = ''
    sarvam_response_file: str = ''
    sarvam_response_source_sha256: str = ''
    upload_authorized: bool = False  # CLI-only; never granted by TOML
    request_timeout: float = 120.0
    ollama_endpoint: str = 'http://127.0.0.1:11434'
    ollama_options: dict = field(default_factory=lambda: {'temperature': 0.0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 2048})
    translation_prompt: str = ''
    translation_group_segments: int = 8
    translation_context_segments: int = 1
    translation_max_chars: int = 4000
    translation_retries: int = 2
    audio_codec: str = 'wav'
    audio_sample_rate: int = 16000
    retain_intermediates: bool = False
    recognition_mode: str = 'codemix'
    translation_mode: str = 'translate'
    output_directory: str = '{name}'
    sarvam_key_env: str = 'SARVAM_API_KEY'
    sarvam_timestamps: bool = True
    sarvam_timestamp_policy: str = 'strict'
    sarvam_read_retries: int = 2
    sarvam_translation_job_id: str = ''
    sarvam_translation_response_file: str = ''


SCHEMA = {
    'paths': {'input': 'input', 'output_dir': 'output_dir'},
    'recognition': {'backend': 'recognition_backend', 'model': 'model', 'language': 'language', 'vocabulary': 'vocabulary', 'prompt': 'prompt', 'mode': 'recognition_mode'},
    'translation': {'backend': 'translation_backend', 'model': 'translation_model', 'prompt': 'translation_prompt',
                    'endpoint': 'ollama_endpoint', 'options': 'ollama_options', 'group_segments': 'translation_group_segments',
                    'context_segments': 'translation_context_segments', 'max_chars': 'translation_max_chars', 'retries': 'translation_retries', 'mode': 'translation_mode'},
    'outputs': {'transcript': 'transcript', 'translation': 'translation', 'formats': 'formats', 'filename': 'filename', 'directory': 'output_directory'},
    'batch': {'recursive': 'recursive', 'continue_on_error': 'continue_on_error'},
    'runtime': {'overwrite': 'overwrite', 'offline': 'offline', 'request_timeout': 'request_timeout'},
    'preprocessing': {'enabled': 'preprocess_enabled', 'normalize': 'normalize_audio', 'vad_backend': 'vad_backend',
                      'threshold': 'vad_threshold', 'min_silence_seconds': 'vad_min_silence', 'padding_seconds': 'vad_padding',
                      'min_speech_ms': 'vad_min_speech_ms', 'max_intervals': 'vad_max_intervals',
                      'codec': 'audio_codec', 'sample_rate': 'audio_sample_rate', 'retain_intermediates': 'retain_intermediates'},
    'sarvam': {'diarization': 'sarvam_diarization', 'with_diarization': 'sarvam_diarization',
               'num_speakers': 'sarvam_num_speakers', 'speaker_names': 'sarvam_speaker_names', 'budget_inr': 'sarvam_budget', 'budget_scope': 'sarvam_budget_scope', 'rate_per_minute': 'sarvam_rate',
               'diarization_rate_per_minute': 'sarvam_diarization_rate', 'cost_margin': 'sarvam_cost_margin',
               'poll_interval': 'sarvam_poll_interval', 'poll_timeout': 'sarvam_poll_timeout',
               'response_file': 'sarvam_response_file', 'response_source_sha256': 'sarvam_response_source_sha256',
               'translation_response_file': 'sarvam_translation_response_file', 'api_key_env': 'sarvam_key_env',
               'with_timestamps': 'sarvam_timestamps', 'timestamp_policy': 'sarvam_timestamp_policy', 'read_retries': 'sarvam_read_retries'},
}


def output_name(c, source, stage):
    try:
        name = c.filename.format(stem=source.stem, extension=source.suffix[1:], stage=stage, mode=stage_mode(c, stage))
    except (KeyError, ValueError, IndexError, AttributeError) as e:
        raise PipelineError(f'Invalid outputs.filename: {e}') from e
    if not name or name in {'.', '..'} or any(x in name for x in ('/', '\\', '\0', '\n', '\r')):
        raise PipelineError('outputs.filename must produce a single safe filename')
    if name + '.json' == 'manifest.json' or name == '.pipeline':
        raise PipelineError('outputs.filename uses a reserved name')
    return name


def stage_mode(c, stage):
    if stage == 'transcript':
        return c.recognition_mode if c.recognition_backend == 'sarvam' else 'transcribe'
    return c.translation_mode


def output_path(c, source, relative):
    try:
        name = c.output_directory.format(stem=source.stem, name=source.name, extension=source.suffix[1:])
    except (KeyError, ValueError, AttributeError, IndexError) as e:
        raise PipelineError(f'Invalid outputs.directory: {e}') from e
    if not name or name in {'.', '..'} or any(ch in name for ch in ('/', '\\', '\0', '\n', '\r')):
        raise PipelineError('outputs.directory must produce one safe directory name')
    return Path(c.output_dir) / relative.parent / name


def validate(c):
    defaults = Config()
    for key, value in asdict(c).items():
        if key == 'sarvam_num_speakers':
            if value is not None and (type(value) is not int or not 1 <= value <= 20):
                raise PipelineError('sarvam.num_speakers must be an integer from 1 to 20, or omitted for automatic detection')
            continue
        expected = type(getattr(defaults, key))
        if expected is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise PipelineError(f'{key} must be a finite number')
            setattr(c, key, float(value))
            continue
        if type(value) is not expected:
            raise PipelineError(f'{key} must be {expected.__name__}')
        if isinstance(value, str) and (not value.strip() and key not in {'prompt', 'translation_prompt', 'sarvam_job_id', 'sarvam_translation_job_id', 'sarvam_response_file', 'sarvam_translation_response_file', 'sarvam_response_source_sha256'}):
            raise PipelineError(f'{key} must not be empty')
    if not c.transcript and not c.translation:
        raise PipelineError('Enable at least one output')
    if not c.formats or any(type(x) is not str or x not in {'txt', 'srt', 'json'} for x in c.formats):
        raise PipelineError('outputs.formats must contain txt, srt and/or json')
    if len(c.formats) != len(set(c.formats)):
        raise PipelineError('Duplicate output formats')
    if any(type(x) is not str or not x.strip() for x in c.vocabulary):
        raise PipelineError('recognition.vocabulary must be an array of nonempty strings')
    # The list is bundled with MLX; configuration tests do not import MLX.
    languages = ('en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue').split()
    if c.language != 'auto' and c.language not in languages and not (c.recognition_backend == 'sarvam' and c.language in {'unknown', 'ta-IN', 'en-IN', 'hi-IN', 'bn-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'od-IN', 'pa-IN', 'te-IN', 'gu-IN', 'as-IN', 'ur-IN', 'ne-IN', 'kok-IN', 'ks-IN', 'sd-IN', 'sa-IN', 'sat-IN', 'mni-IN', 'brx-IN', 'mai-IN', 'doi-IN'}):
        raise PipelineError('recognition.language must be a Whisper language code or auto')
    if c.recognition_backend not in {'mlx_whisper', 'sarvam'} or c.translation_backend not in {'mlx_whisper', 'ollama', 'sarvam'}:
        raise PipelineError('Recognition supports mlx_whisper/sarvam; translation supports mlx_whisper/ollama/sarvam')
    if c.audio_codec not in {'wav', 'flac'} or c.audio_sample_rate not in {8000, 16000, 24000, 48000}:
        raise PipelineError('Audio codec must be wav/flac; sample rate must be 8000/16000/24000/48000')
    if c.vad_backend == 'silero' and c.audio_sample_rate not in {8000, 16000}:
        raise PipelineError('Silero requires 8000 or 16000 Hz audio')
    if c.recognition_mode != 'codemix' or c.translation_mode != 'translate':
        raise PipelineError('Recognition mode must be codemix and translation mode must be translate')
    import re
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', c.sarvam_key_env):
        raise PipelineError('sarvam.api_key_env must name a process environment variable, never contain a key')
    if c.sarvam_timestamp_policy not in {'strict', 'flag'} or not 0 <= c.sarvam_read_retries <= 5:
        raise PipelineError('Invalid Sarvam timestamp policy or read retry count')
    if c.translation_backend == 'sarvam':
        if c.translation_model == LARGE:
            c.translation_model = 'saaras:v4'
        if c.translation_model != 'saaras:v4' or c.translation_prompt:
            raise PipelineError('Sarvam translation requires saaras:v4; text-generation prompts are not supported')
    if c.translation and c.translation_backend == 'mlx_whisper' and c.translation_model != LARGE:
        path = Path(c.translation_model).expanduser()
        if not path.is_dir():
            raise PipelineError('Translation requires multilingual Large V3, not Turbo; use the default repo or a local Large V3 directory')
        verify_large(path)
    if c.transcript and (c.model.endswith('.en') or '.en-' in c.model) and c.language != 'en':
        raise PipelineError('An English-only model cannot transcribe Tamil/mixed audio')
    if c.vad_backend not in {'none', 'silero'} or (c.vad_backend != 'none' and not c.preprocess_enabled):
        raise PipelineError('Silero VAD requires preprocessing.enabled=true; use vad_backend="none" to disable VAD')
    if (not 0 < c.vad_threshold < 1 or c.vad_min_silence < 1 or c.vad_padding < 0.1
            or c.vad_min_speech_ms < 1 or c.vad_max_intervals < 1):
        raise PipelineError('Invalid VAD thresholds/padding; retain at least 0.1 seconds of boundary padding')
    if c.sarvam_num_speakers is not None and not c.sarvam_diarization:
        raise PipelineError('sarvam.num_speakers requires diarization=true')
    if c.sarvam_diarization and not c.sarvam_timestamps:
        raise PipelineError('Speaker-attributed outputs require with_timestamps=true')
    if any(mode not in {'codemix', 'translate'} or not isinstance(names, dict)
           or any(not isinstance(key, str) or not key.strip() or not isinstance(name, str)
                  or not name.strip() or any(ord(ch) < 32 for ch in key + name)
                  for key, name in names.items()) for mode, names in c.sarvam_speaker_names.items()):
        raise PipelineError('sarvam.speaker_names must map codemix/translate to provider-ID/name tables without control characters')
    if c.sarvam_budget_scope not in {'persistent', 'recording', 'invocation'}:
        raise PipelineError('sarvam.budget_scope must be persistent, recording, or invocation')
    if (c.sarvam_budget < 0 or c.sarvam_rate <= 0 or c.sarvam_diarization_rate <= 0
            or c.sarvam_cost_margin < 1 or c.sarvam_poll_interval < 0.1
            or c.sarvam_poll_interval > 30 or c.sarvam_poll_timeout <= 0 or c.request_timeout <= 0):
        raise PipelineError('Invalid provider budget/rates/timeouts (poll interval must be 0.1–30 seconds)')
    if c.recognition_backend == 'sarvam' or c.translation_backend == 'sarvam':
        if (c.recognition_backend == 'sarvam' and c.model != 'saaras:v4') or len(set(c.vocabulary)) > 50 or any(len(term) > 64 for term in c.vocabulary):
            raise PipelineError('Sarvam requires saaras:v4 and at most 50 distinct keyterms of at most 64 characters')
        allowed = {'unknown', 'hi', 'bn', 'kn', 'ml', 'mr', 'od', 'pa', 'ta', 'te', 'en', 'gu', 'as', 'ur', 'ne', 'kok', 'ks', 'sd', 'sa', 'sat', 'mni', 'brx', 'mai', 'doi'}
        if c.language != 'auto' and c.language.removesuffix('-IN') not in allowed:
            raise PipelineError('Unsupported Saaras language')
        if c.recognition_backend == 'sarvam' and c.prompt:
            raise PipelineError('Saaras codemix uses recognition.vocabulary as keyterms; recognition.prompt is only for MLX')
        if len(c.vocabulary) != len(set(c.vocabulary)):
            raise PipelineError('Saaras keyterms must be distinct')
    if c.sarvam_response_file or c.sarvam_translation_response_file:
        if (c.recognition_backend != 'sarvam' or c.vad_backend != 'none'
                or len(c.sarvam_response_source_sha256) != 64
                or any(ch not in '0123456789abcdef' for ch in c.sarvam_response_source_sha256)
                or any(not Path(path).is_file() for path in (c.sarvam_response_file, c.sarvam_translation_response_file) if path)):
            raise PipelineError('Import requires an existing Saaras response_file, exact lowercase source SHA-256, and vad_backend="none"')
    for job_id in (c.sarvam_job_id, c.sarvam_translation_job_id):
        if job_id and not re.fullmatch(r'[A-Za-z0-9_-]+', job_id):
            raise PipelineError('Invalid Sarvam job ID')
    if c.translation_backend == 'ollama':
        from .ollama_provider import validate_endpoint
        validate_endpoint(c.ollama_endpoint)
        if c.translation_model == LARGE:
            c.translation_model = 'gemma3:12b'
        if (c.translation_group_segments < 1 or c.translation_context_segments < 0
                or c.translation_max_chars < 100 or not 0 <= c.translation_retries <= 5):
            raise PipelineError('Invalid bounded translation group/retry configuration')
        allowed_options = {'temperature', 'seed', 'num_ctx', 'num_predict', 'top_p', 'top_k', 'repeat_penalty'}
        if set(c.ollama_options) - allowed_options or any(type(v) not in (int, float) or not math.isfinite(v) for v in c.ollama_options.values()):
            raise PipelineError('Invalid Ollama generation options')
        if c.ollama_options.get('num_ctx', 8192) < 1024 or c.ollama_options.get('num_predict', 2048) <= 0:
            raise PipelineError('Ollama context/output token limits must be bounded and positive')
        for key in ('num_ctx', 'num_predict', 'seed', 'top_k'):
            if key in c.ollama_options and type(c.ollama_options[key]) is not int:
                raise PipelineError(f'Ollama {key} must be an integer')
        if not 0 <= c.ollama_options.get('temperature', 0) <= 2 or not 0 <= c.ollama_options.get('top_p', 1) <= 1:
            raise PipelineError('Invalid Ollama temperature/top_p range')
    try:
        for _, key, spec, conversion in string.Formatter().parse(c.filename):
            if key is not None and (key not in {'stem', 'extension', 'stage', 'mode'} or spec or conversion):
                raise PipelineError('filename allows {stem}, {extension}, {stage}, {mode}, with no formatting modifiers')
    except ValueError as e:
        raise PipelineError(f'Invalid outputs.filename: {e}') from e
    if '{stage}' not in c.filename and '{mode}' not in c.filename:
        raise PipelineError('outputs.filename must include {stage} or {mode} to separate outputs')
    try:
        for _, key, spec, conversion in string.Formatter().parse(c.output_directory):
            if key is not None and (key not in {'stem', 'name', 'extension'} or spec or conversion):
                raise PipelineError('Invalid outputs.directory placeholder')
    except ValueError as e:
        raise PipelineError(f'Invalid outputs.directory: {e}') from e
    output_path(c, Path('தமிழ் (1).mov'), Path('தமிழ் (1).mov'))
    for stage in STAGES:
        output_name(c, Path('தமிழ் recording (1).mov'), stage)
    return c


def verify_large(path):
    try:
        dims = json.loads((path / 'config.json').read_text())
        expected = {'n_mels': 128, 'n_vocab': 51866, 'n_audio_layer': 32, 'n_text_layer': 32,
                    'n_audio_state': 1280, 'n_text_state': 1280}
        if any(dims.get(k) != v for k, v in expected.items()):
            raise ValueError('architecture does not match multilingual Large V3')
    except (OSError, ValueError, TypeError, AttributeError) as e:
        raise PipelineError(f'Invalid translation model at {path}: {e}') from e


def parser():
    p = argparse.ArgumentParser(prog='mozhiflow', description=__doc__)
    from . import __version__
    p.add_argument('--version', action='version', version=f'MozhiFlow {__version__}')
    p.add_argument('input', nargs='?', help='Media file or batch directory')
    p.add_argument('--config', type=Path, help='Default: config.toml in the source checkout, or current directory when installed')
    for flag in ('output-dir', 'model', 'language', 'prompt', 'recognition-backend', 'translation-backend', 'translation-model', 'translation-prompt', 'ollama-endpoint', 'sarvam-job-id', 'sarvam-translation-job-id', 'filename', 'output-directory', 'audio-codec', 'sarvam-key-env', 'sarvam-timestamp-policy'):
        p.add_argument('--' + flag, default=argparse.SUPPRESS)
    p.add_argument('--vocabulary', nargs='+', default=argparse.SUPPRESS)
    p.add_argument('--formats', nargs='+', choices=['txt', 'srt', 'json'], default=argparse.SUPPRESS)
    for flag in ('transcript', 'translation', 'recursive', 'continue-on-error', 'overwrite', 'offline', 'retain-intermediates'):
        p.add_argument('--' + flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    p.add_argument('--only', choices=STAGES, help='Run only this stage, including retries')
    p.add_argument('--force', dest='overwrite', action='store_true', default=argparse.SUPPRESS)
    p.add_argument('--fast', dest='model', action='store_const', const=TURBO, default=argparse.SUPPRESS)
    p.add_argument('--dry-run', action='store_true', help='Validate and print plan without creating files or loading MLX')
    p.add_argument('--allow-upload', dest='upload_authorized', action='store_true', default=argparse.SUPPRESS,
                   help='Explicitly authorize audio upload for the selected input(s); requires configured budget')
    p.add_argument('--sarvam-budget-scope', choices=['persistent', 'recording', 'invocation'], default=argparse.SUPPRESS)
    p.add_argument('--sarvam-budget', type=float, default=argparse.SUPPRESS, help='Local cost ceiling in INR for sarvam.budget_scope')
    p.add_argument('--audio-sample-rate', type=int, default=argparse.SUPPRESS)
    return p


def load_config(args):
    c = Config()
    config_path = args.config if args.config is not None else default_config_root / 'config.toml'
    try:
        with config_path.open('rb') as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        # A missing explicitly selected config is always an error.
        if args.config is not None:
            raise PipelineError(f'Configuration not found: {config_path}')
        data = {}
    except (OSError, tomllib.TOMLDecodeError, UnicodeError) as e:
        raise PipelineError(f'Cannot read configuration: {e}') from e
    sarvam = data.get('sarvam', {})
    if isinstance(sarvam, dict) and 'diarization' in sarvam and 'with_diarization' in sarvam and sarvam['diarization'] != sarvam['with_diarization']:
        raise PipelineError('Conflicting sarvam.diarization and sarvam.with_diarization settings')
    for section, entries in data.items():
        if section not in SCHEMA or not isinstance(entries, dict):
            raise PipelineError(f'Unknown or invalid configuration section: {section}')
        for key, value in entries.items():
            if key not in SCHEMA[section]:
                raise PipelineError(f'Unknown configuration key: {section}.{key}')
            setattr(c, SCHEMA[section][key], value)
    base = config_path.resolve().parent
    # Config paths are relative to the config; CLI paths are relative to cwd.
    for key in ('input', 'output_dir', 'model', 'translation_model', 'sarvam_response_file', 'sarvam_translation_response_file'):
        value = getattr(c, key)
        if isinstance(value, str) and "\0" in value:
            raise PipelineError(f"{key} contains a null character")
        if isinstance(value, str) and value and (key in {'input', 'output_dir', 'sarvam_response_file', 'sarvam_translation_response_file'} or value.startswith(('.', '/', '~'))):
            setattr(c, key, str((base / Path(value).expanduser()).resolve()))
    for key in asdict(c):
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(c, key, getattr(args, key))
    for key in ('input', 'output_dir', 'model', 'translation_model', 'sarvam_response_file', 'sarvam_translation_response_file'):
        value = getattr(c, key)
        if isinstance(value, str) and '\0' in value:
            raise PipelineError(f'{key} contains a null character')
        if isinstance(value, str) and value and (key in {'input', 'output_dir', 'sarvam_response_file', 'sarvam_translation_response_file'} or value.startswith(('.', '/', '~'))):
            setattr(c, key, str(Path(value).expanduser().resolve()))
    if args.only:
        c.transcript = args.only == 'transcript'
        c.translation = args.only == 'translation'
    return validate(c)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def identity(path):
    before = path.stat()
    sha = digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PipelineError(f'Source changed while hashing: {path}')
    return {'path': str(path.resolve()), 'size': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': sha}


def discover(c):
    source, out = Path(c.input).resolve(), Path(c.output_dir).resolve()
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            raise PipelineError(f'Unsupported media extension: {source}')
        return [(source, Path(source.name))]
    if not source.is_dir():
        raise PipelineError(f'Input does not exist or is not a file/directory: {source}')
    if source == out or source.is_relative_to(out):
        raise PipelineError('Batch input must not be inside the output directory')
    files = source.rglob('*') if c.recursive else source.glob('*')
    result = [(p, p.relative_to(source)) for p in files if p.is_file() and not p.is_symlink()
              and p.suffix.lower() in EXTENSIONS and not p.resolve().is_relative_to(out)]
    if not result:
        raise PipelineError(f'No supported media found in {source}')
    return sorted(result)


def stage_settings(c, stage):
    result = {'stage': stage, 'task': 'transcribe' if stage == 'transcript' else 'translate',
            'model': c.model if stage == 'transcript' else c.translation_model,
            'backend': 'mlx_whisper' if stage == 'transcript' else c.translation_backend,
            'language': c.language, 'vocabulary': c.vocabulary, 'prompt': c.prompt,
            'formats': c.formats, 'filename': c.filename, 'pipeline_version': 1}
    if stage == 'transcript' and c.recognition_backend == 'sarvam':
        result.update(backend='sarvam', task='codemix', diarization=c.sarvam_diarization)
        if c.sarvam_response_file:
            result['imported_response_sha256'] = digest(Path(c.sarvam_response_file))
            result['imported_source_sha256'] = c.sarvam_response_source_sha256
    if stage == 'translation' and c.translation_backend == 'sarvam':
        result.update(task=c.translation_mode, backend='sarvam', model=c.translation_model,
                      diarization=c.sarvam_diarization, prompt='')
        if c.sarvam_translation_response_file:
            result['imported_response_sha256'] = digest(Path(c.sarvam_translation_response_file))
            result['imported_source_sha256'] = c.sarvam_response_source_sha256
    if result['backend'] == 'sarvam':
        if c.sarvam_num_speakers is not None:
            result['num_speakers'] = c.sarvam_num_speakers
        names = c.sarvam_speaker_names.get(result['task'], {})
        if names:
            result['speaker_names'] = names
        if not c.sarvam_timestamps:
            result['with_timestamps'] = False
        if c.sarvam_timestamp_policy != 'strict':
            result['timestamp_policy'] = c.sarvam_timestamp_policy
    if c.preprocess_enabled or result['backend'] == 'sarvam':
        from .audio_preprocess import settings
        result['preprocessing'] = settings(c)
    if stage == 'translation' and c.translation_backend == 'ollama':
        result.update(task='translate_text', prompt=c.translation_prompt, endpoint=c.ollama_endpoint,
                      options=c.ollama_options, group_segments=c.translation_group_segments,
                      context_segments=c.translation_context_segments, max_chars=c.translation_max_chars)
    return result


def source_transcript(directory, manifest, c, src):
    stage = manifest['stages'].get('transcript', {})
    path = directory / '.inference' / 'transcript.json'
    cache = stage.get('inference_checkpoint', {})
    if not path.is_file() or cache.get('sha256') != digest(path):
        raise PipelineError('Local text translation requires an intact transcript checkpoint; run the transcript stage first (no upload will be attempted by translation)')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('source') != src or data.get('settings') != stage_settings(c, 'transcript'):
        raise PipelineError('Transcript checkpoint does not match recognition/source settings; run transcript with matching settings first')
    return data['result']


def inference(source, settings, offline):
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    else:
        os.environ.pop('HF_HUB_OFFLINE', None)
    try:
        from huggingface_hub import snapshot_download
        import mlx_whisper
    except ImportError as e:
        if isinstance(e, ModuleNotFoundError):
            raise PipelineError('MLX Whisper dependency is missing. Run ./setup.sh') from e
        raise PipelineError(f'Cannot initialize MLX Whisper: {e}') from e
    model = settings['model']
    model_path = Path(model).expanduser()
    if not model_path.is_dir():
        # Explicit local_files_only ensures cache-only processing even in an imported process.
        model_path = Path(snapshot_download(model, local_files_only=offline))
    if settings['stage'] == 'translation':
        verify_large(model_path)
    prompt = '\n'.join(x for x in (settings['prompt'], ', '.join(settings['vocabulary'])) if x)
    return mlx_whisper.transcribe(str(source), path_or_hf_repo=str(model_path),
                                 task=settings['task'], language=None if settings['language'] == 'auto' else settings['language'],
                                 initial_prompt=prompt or None, verbose=None)


def isolated_inference(source, settings, offline):
    """A native MLX crash must not abort the orchestrator or another stage."""
    with tempfile.TemporaryDirectory(prefix='whisper-worker-') as tmp:
        result_path = Path(tmp) / 'result.json'
        request = {'source': str(source), 'settings': settings, 'offline': offline,
                   'result_path': str(result_path)}
        worker = subprocess.run([sys.executable, str(Path(__file__).with_name('_worker.py'))],
                                input=json.dumps(request), text=True, capture_output=True)
        if worker.returncode:
            detail = (worker.stderr or worker.stdout).strip()[-4000:]
            raise PipelineError(f'Inference worker exited {worker.returncode}: {detail or "no diagnostic output"}')
        try:
            return json.loads(result_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            raise PipelineError(f'Inference worker produced no valid result: {e}') from e


def worker_main():
    try:
        request = json.load(sys.stdin)
        result = inference(Path(request['source']), request['settings'], request['offline'])
        result = prepare_result(result)
        Path(request['result_path']).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return 0
    except Exception as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr)
        return 1


def timestamp_problem(segment):
    a, b = segment.get('start'), segment.get('end')
    if (type(a) not in (int, float) or type(b) not in (int, float)
            or not math.isfinite(a) or not math.isfinite(b) or not 0 <= a <= b
            or (a == b and segment['text'].strip())):
        return (f'start={a!r} ({type(a).__name__}), end={b!r} ({type(b).__name__}); '
                'expected finite, nonnegative times with end > start for spoken text')
    return None


def validate_result(result, *, strict_timestamps=True):
    if not isinstance(result, dict) or not isinstance(result.get('text'), str) or not result['text'].strip():
        raise PipelineError('Whisper returned empty or invalid text')
    segments = result.get('segments')
    if not isinstance(segments, list) or not segments:
        raise PipelineError('Whisper returned no timestamped segments')
    spoken_segments = 0
    for index, s in enumerate(segments):
        if not isinstance(s, dict) or not isinstance(s.get('text'), str):
            raise PipelineError('Invalid Whisper segment text')
        problem = timestamp_problem(s)
        if strict_timestamps and problem:
            raise PipelineError(f'Invalid Whisper segment timestamps at segments[{index}]: '
                                f'{problem}')
        if s['text'].strip():
            spoken_segments += 1
    if not spoken_segments:
        raise PipelineError('Whisper returned no nonempty timestamped speech segments')


def prepare_result(result):
    """Retain usable speech despite faulty timing; never fabricate replacement times."""
    validate_result(result, strict_timestamps=False)
    issues = [{'segment_index': index, 'start': repr(segment.get('start')),
               'end': repr(segment.get('end')), 'has_text': bool(segment['text'].strip()),
               'reason': timestamp_problem(segment)}
              for index, segment in enumerate(result['segments']) if timestamp_problem(segment)]
    replacements = []

    def clean(value, path='$'):
        if isinstance(value, float) and not math.isfinite(value):
            replacements.append({'path': path, 'original': str(value)})
            return None
        if isinstance(value, dict):
            return {key: clean(item, f'{path}.{key}') for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item, f'{path}[{index}]') for index, item in enumerate(value)]
        return value

    prepared = clean(result)
    if replacements:
        prepared['_serialization'] = {'non_finite_values': replacements,
                                      'encoding': 'Non-finite numeric values represented as JSON null'}
    if issues and '_timing' not in prepared:
        prepared['_timing'] = {'invalid_segments': issues,
                               'srt_complete': not any(issue['has_text'] for issue in issues),
                               'policy': 'Preserve all text and JSON segments; omit invalid cues from SRT without guessing times'}
    return prepared


def timestamp(seconds):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'


def render(result, fmt):
    if fmt == 'txt':
        if result.get('diarization'):
            return ''.join(f"[{timestamp(s['start'])} --> {timestamp(s['end'])}] {s['speaker_label']}: {s['text']}\n"
                           if not timestamp_problem(s) else f"[timestamp unavailable] {s['speaker_label']}: {s['text']}\n"
                           for s in result['segments'])
        return result['text']  # Keep the raw model text, including repetitions and whitespace.
    if fmt == 'json':
        return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    return ''.join(f"{i}\n{timestamp(s['start'])} --> {timestamp(s['end'])}\n{(s['speaker_label'] + ': ') if result.get('diarization') else ''}{s['text']}\n\n"
                   for i, s in enumerate((s for s in result['segments']
                                          if s['text'].strip() and not timestamp_problem(s)), 1))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path, data):
    # Only our manifest is replaced; original media and completed outputs are never opened for writing here.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.manifest-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


@contextmanager
def locked(directory):
    with (directory / '.pipeline.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise PipelineError(f'Another process is handling {directory}') from e
        yield


def intact(record, directory):
    artifacts = record.get('artifacts', {})
    return bool(artifacts) and all((directory / n).is_file() and digest(directory / n) == h
                                   for n, h in artifacts.items())


def process(source, relative, c, run_inference=isolated_inference):
    import uuid
    invocation = getattr(c, '_budget_invocation', '') or uuid.uuid4().hex
    # Keeping the full original filename in the directory distinguishes meeting.mov from meeting.wav.
    directory = output_path(c, source, relative)
    if not directory.resolve().is_relative_to(Path(c.output_dir).resolve()):
        raise PipelineError(f'Output directory escapes configured root: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    with locked(directory):
        manifest_path = directory / 'manifest.json'
        src = identity(source)
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                if manifest.get('version') != 1 or not isinstance(manifest.get('stages'), dict):
                    raise ValueError('unsupported manifest schema')
            except (ValueError, OSError, AttributeError) as e:
                raise PipelineError(f'Cannot read manifest {manifest_path}: {e}') from e
            if manifest['source'] != src:
                raise PipelineError(f'Source identity changed for {source}; choose a new output directory')
        else:
            manifest = {'version': 1, 'source': src, 'created_at': now(), 'stages': {}}
        manifest['configuration'] = asdict(c)

        def save():
            write_manifest(manifest_path, manifest)

        prepared_audio = None

        def selected_cloud_modes():
            return [stage_mode(c, stage) for stage in STAGES if getattr(c, stage)
                    and (c.recognition_backend if stage == 'transcript' else c.translation_backend) == 'sarvam']

        def grant_key(prep):
            from .toolkit_core import fingerprint
            grant = {'audio': prep['audio_sha256'], 'language': c.language,
                     'keyterms': c.vocabulary, 'diarization': c.sarvam_diarization, 'model': 'saaras:v4'}
            if c.sarvam_num_speakers is not None: grant['num_speakers'] = c.sarvam_num_speakers
            return fingerprint(grant)

        def cloud_config(mode, prep):
            grant = manifest.get('upload_authorizations', {}).get(grant_key(prep), {})
            return replace(c, upload_authorized=c.upload_authorized or mode in grant.get('modes', []))

        def cloud_budget(prep):
            from .sarvam_provider import budget_preflight
            jobs = cloud_jobs(c, source, relative, src, prep)
            return budget_preflight(c, jobs, invocation=invocation, reserve=True)

        def audio():
            nonlocal prepared_audio
            if prepared_audio is not None:
                return prepared_audio
            from .audio_preprocess import prepare, settings as audio_settings
            from .toolkit_core import fingerprint
            signature = audio_settings(c)
            work = directory / '.preprocessing' / fingerprint(signature)
            if not work.resolve().is_relative_to(directory.resolve()):
                raise PipelineError('Preprocessing path escapes output directory')
            manifest['preprocessing'] = {'status': 'running', 'configuration': signature}
            save()
            try:
                path, metadata = prepare(source, work, c, src)
                if identity(source) != src:
                    raise PipelineError('Source changed during local preprocessing')
                manifest['preprocessing'] = metadata
                manifest['preprocessing']['audio_path'] = (str(path) if metadata.get('source_audio_reused')
                                                           else str(path.relative_to(directory)))
                if c.upload_authorized:
                    grants = manifest.setdefault('upload_authorizations', {})
                    previous_modes = grants.get(grant_key(metadata), {}).get('modes', [])
                    grants[grant_key(metadata)] = {'audio_sha256': metadata['audio_sha256'],
                        'modes': sorted(set(previous_modes + selected_cloud_modes())), 'authorized_at': now()}
                save()
                prepared_audio = (path, metadata)
                return prepared_audio
            except Exception as e:
                manifest['preprocessing'].update(status='failed', error=f'{type(e).__name__}: {e}')
                save()
                raise

        if selected_cloud_modes():
            from .sarvam_provider import budget_preflight, BudgetPreflightError
            _, prep = audio()
            jobs = cloud_jobs(c, source, relative, src, prep)
            report = budget_preflight(c, jobs, invocation=invocation) if prep['retained_duration'] <= 7200 else {'processing_permitted': True, 'note': 'Duration exceeds Batch limit; no reservation made'}
            manifest['budget_preflight'] = report
            save()
            print('PREFLIGHT ' + json.dumps(report, ensure_ascii=False))
            if not report['processing_permitted']:
                raise BudgetPreflightError('Local Sarvam budget preflight blocked; no new paid submissions; see budget breakdown')

        counts = {'completed': 0, 'skipped': 0, 'failed': 0}
        for stage in STAGES:
            if not getattr(c, stage):
                continue
            settings = stage_settings(c, stage)
            previous = manifest['stages'].get(stage, {})
            transcript = None
            if stage == 'translation' and c.translation_backend == 'ollama':
                try:
                    from .toolkit_core import fingerprint
                    transcript = source_transcript(directory, manifest, c, src)
                    settings['source_transcript_sha256'] = fingerprint(transcript)
                except Exception as e:
                    if previous.get('status') != 'success':
                        manifest['stages'][stage] = {'status': 'failed', 'settings': settings, 'error': str(e),
                                                     'configuration': asdict(c), 'attempts': previous.get('attempts', 0)}
                        save()
                    print(f'ERROR {stage} ({source}): {e}', file=sys.stderr)
                    counts['failed'] += 1
                    if not c.continue_on_error:
                        break
                    continue
            name = output_name(c, source, stage)
            targets = {f'{name}.{fmt}': fmt for fmt in c.formats}
            if (not c.overwrite and previous.get('status') == 'success' and previous.get('settings') == settings
                    and set(previous.get('artifacts', {})) == set(targets) and intact(previous, directory)):
                print(f'SKIP {stage}: {source}')
                if previous.get('timing'):
                    print(f'WARNING {stage}: saved outputs have timing issues; see manifest.json and JSON _timing.', file=sys.stderr)
                counts['skipped'] += 1
                continue
            try:
                # Retry may replace files from its own failed attempt only if still byte-for-byte intact.
                owned_targets = {}
                for filename in targets:
                    target = directory / filename
                    if target.is_symlink():
                        raise PipelineError(f'Refusing symlink output: {target}')
                    if target.exists() and not c.overwrite:
                        owned = (previous.get('status') in {'failed', 'running'} and previous.get('settings') == settings
                                 and previous.get('artifacts', {}).get(filename) == digest(target))
                        if not owned:
                            raise PipelineError(f'Output already exists: {target}; use --overwrite or a new output directory')
                        owned_targets[filename] = previous['artifacts'][filename]
                if previous.get('status') == 'success' and not c.overwrite:
                    raise PipelineError(f'{stage} settings or outputs changed; use --overwrite or a new output directory')
                record = {'status': 'running', 'settings': settings, 'configuration': asdict(c),
                          'started_at': now(), 'attempts': previous.get('attempts', 0) + 1,
                          'artifacts': dict(previous.get('artifacts', {}))}
                manifest['stages'][stage] = record
                write_manifest(manifest_path, manifest)
                print(f'RUN {stage}: {source}', flush=True)
                if stage == 'translation' and c.translation_backend == 'mlx_whisper':
                    print(f'WARNING: {WARNING}', file=sys.stderr, flush=True)
                try:
                    checkpoint = directory / '.inference' / f'{stage}.json'
                    result = None
                    cached_record = previous.get('inference_checkpoint', {})
                    if (not c.overwrite and checkpoint.is_file() and not checkpoint.is_symlink()
                            and cached_record.get('sha256') == digest(checkpoint)):
                        cached = json.loads(checkpoint.read_text(encoding='utf-8'))
                        if cached.get('source') == src and cached.get('settings') == settings:
                            result = cached['result']
                            record['attempts'] = previous.get('attempts', 0)
                            if 'provider' in previous:
                                record['provider'] = previous['provider']
                            print(f'REUSE {stage} inference checkpoint: {source}', flush=True)
                    if result is None:
                        from .providers import RECOGNITION, TRANSLATION
                        context = {'config': c, 'directory': directory, 'source_identity': src,
                                   'record': record, 'save': save, 'audio': audio, 'inference': run_inference,
                                   'transcript': transcript, 'cloud_config': cloud_config, 'cloud_budget': cloud_budget}
                        if stage == 'transcript':
                            result = RECOGNITION[c.recognition_backend].recognize(source, settings, context)
                        else:
                            result = TRANSLATION[c.translation_backend].translate(source, settings, context)
                    result = prepare_result(result)
                    if result.get('_serialization'):
                        record['serialization'] = result['_serialization']
                        fields = ', '.join(item['path'] for item in result['_serialization']['non_finite_values'])
                        print(f'WARNING {stage}: non-finite model diagnostics stored as JSON null: {fields}. '
                              'Text and finite timestamps are preserved; review recognition quality.', file=sys.stderr, flush=True)
                    if identity(source) != src:
                        raise PipelineError('Source changed during processing; results were not published')
                    # Preserve inference before rendering/publishing so a later output error
                    # does not require decoding a long recording again.
                    if not checkpoint.parent.resolve().is_relative_to(directory.resolve()):
                        raise PipelineError('Inference checkpoint directory escapes output directory')
                    checkpoint.parent.mkdir(exist_ok=True)
                    write_manifest(checkpoint, {'source': src, 'settings': settings, 'result': result})
                    record['inference_checkpoint'] = {'path': str(checkpoint.relative_to(directory)),
                                                      'sha256': digest(checkpoint)}
                    if result.get('_timing'):
                        record['timing'] = result['_timing']
                        issues = result['_timing']['invalid_segments']
                        spoken = sum(issue['has_text'] for issue in issues)
                        print(f'WARNING {stage}: {len(issues)} invalid timestamp segment(s); '
                              f'{spoken} spoken cue(s) omitted from SRT. All text is retained in TXT/JSON. '
                              'Review JSON _timing and manifest.json; no timestamps were guessed.',
                              file=sys.stderr, flush=True)
                    write_manifest(manifest_path, manifest)
                    with tempfile.TemporaryDirectory(dir=directory, prefix=f'.{stage}-') as tmp:
                        for filename, fmt in targets.items():
                            (Path(tmp) / filename).write_text(render(result, fmt), encoding='utf-8')
                        for filename in targets:
                            target, staged = directory / filename, Path(tmp) / filename
                            if c.overwrite or filename in owned_targets:
                                if not c.overwrite and (target.is_symlink() or not target.is_file()
                                                        or digest(target) != owned_targets[filename]):
                                    raise PipelineError(f'Output changed during retry: {target}')
                                os.replace(staged, target)
                            else:
                                os.link(staged, target)  # Atomic no-clobber publication.
                            record['artifacts'][filename] = digest(target)
                            write_manifest(manifest_path, manifest)
                    record.update(status='success', finished_at=now())
                    if stage == 'translation' and c.translation_backend == 'mlx_whisper':
                        record['warning'] = WARNING
                    write_manifest(manifest_path, manifest)
                    counts['completed'] += 1
                except Exception as e:
                    record.update(status='failed', error=f'{type(e).__name__}: {e}', finished_at=now())
                    write_manifest(manifest_path, manifest)
                    raise
            except Exception as e:
                print(f'ERROR {stage} ({source}): {e}', file=sys.stderr, flush=True)
                counts['failed'] += 1
                if not c.continue_on_error:
                    break
        return counts


def preview_audio(source, relative, c, src):
    """Preview native selection, retaining a historical payload when resuming work."""
    from .audio_preprocess import prepare, settings as audio_settings, decisions
    directory = output_path(c, source, relative)
    manifest_path = directory / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        prep = manifest.get('preprocessing', {})
        if (manifest.get('source') == src and prep.get('configuration') == audio_settings(c)
                and prep.get('status') == 'success'):
            paid = any(json.loads(path.read_text()).get('audio_sha256') == prep.get('audio_sha256')
                       and json.loads(path.read_text()).get('phase') not in {'new', 'imported'}
                       for path in directory.glob('.providers/sarvam/*/remote.json'))
            audio_path = Path(prep.get('audio_path', ''))
            if not audio_path.is_absolute(): audio_path = directory / audio_path
            if paid and audio_path.is_file() and digest(audio_path) == prep.get('audio_sha256'):
                if not prep.get('audio_selection_version'):
                    prep.update(decisions(prep.get('input_kind') == 'audio', prep.get('source_audio_reused', False),
                                          'Retaining cached payload for existing paid-job recovery'))
                return prep
    with tempfile.TemporaryDirectory(prefix='whisper-audio-plan-') as tmp:
        return prepare(source, Path(tmp), c, src)[1]


def cloud_jobs(c, source, relative, src, prep):
    from .sarvam_provider import estimate, job_token
    jobs = []
    directory = output_path(c, source, relative)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for stage in STAGES:
        if not getattr(c, stage) or (c.recognition_backend if stage == 'transcript' else c.translation_backend) != 'sarvam':
            continue
        previous = manifest.get('stages', {}).get(stage, {})
        requested = stage_settings(c, stage)
        old_settings = previous.get('settings', {})
        old_provider = previous.get('provider', {})
        old_path = directory / '.providers' / 'sarvam' / job_token(src, old_settings, prep) / 'remote.json'
        old_remote = json.loads(old_path.read_text()) if old_path.is_file() else {}
        protected = (previous.get('status') == 'success' or old_provider.get('remote_job_id')
                     or old_provider.get('phase') in {'creating', 'started', 'starting'}
                     or old_remote.get('job_id') or old_remote.get('phase') == 'creating')
        changed = any(old_settings.get(key, False if key == 'diarization' else None) != requested.get(key, False if key == 'diarization' else None)
                      for key in ('diarization', 'num_speakers'))
        if protected and old_settings.get('backend') == 'sarvam' and changed:
            raise PipelineError(f'{stage}: diarization settings differ from successful/submitted work. '
                                'No replacement job will be created. Restore its settings to resume; '
                                'preview a separate output directory and explicitly authorize any new diarized jobs')
        if c.sarvam_response_file if stage == 'transcript' else c.sarvam_translation_response_file:
            continue
        token = job_token(src, requested, prep)
        path = directory / '.providers' / 'sarvam' / token / 'remote.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        jobs.append({'token': token, 'recording': src['sha256'], 'mode': stage_mode(c, stage),
                     'cost': estimate(prep['retained_duration'], c)['estimated_cost'],
                     'recoverable': bool(state.get('job_id') or state.get('phase') in {'creating', 'imported', 'downloaded'})})
    return jobs


def main(argv=None):
    args = parser().parse_args(argv)
    from .sarvam_provider import BudgetPreflightError
    try:
        if sys.version_info[:2] != (3, 12):
            raise PipelineError('Python 3.12 is required. Run ./setup.sh')
        c = load_config(args)
        files = discover(c)
        destinations = [output_path(c, source, relative).resolve() for source, relative in files]
        if len(set(destinations)) != len(destinations):
            raise PipelineError('Multiple inputs would share an output directory; use outputs.directory="{name}" or separate inputs')
        import uuid
        c._budget_invocation = uuid.uuid4().hex
        all_jobs = []
        cloud_enabled = any(getattr(c, stage) and (c.recognition_backend if stage == 'transcript' else c.translation_backend) == 'sarvam' for stage in STAGES)
        if args.dry_run or cloud_enabled:
            audio_plans = []
            if c.preprocess_enabled or c.recognition_backend == 'sarvam' or c.translation_backend == 'sarvam':
                from .audio_preprocess import prepare
                from .sarvam_provider import estimate
                for source, relative in files:
                    before = identity(source)
                    if args.dry_run:
                        metadata = preview_audio(source, relative, c, before)
                    else:
                        from .audio_preprocess import settings as audio_settings
                        from .toolkit_core import fingerprint
                        work = output_path(c, source, relative) / '.preprocessing' / fingerprint(audio_settings(c))
                        output_path(c, source, relative).mkdir(parents=True, exist_ok=True)
                        with locked(output_path(c, source, relative)):
                            _, metadata = prepare(source, work, c, before)
                    if identity(source) != before:
                        raise PipelineError('Source changed during audio dry-run')
                    plan = {key: metadata[key] for key in ('original_duration', 'audio_duration', 'retained_duration',
                            'removed_duration', 'predicted_savings_percent', 'no_speech_fallback', 'mapping')}
                    plan['source'] = str(source)
                    plan['trimmed_duration'] = metadata['retained_duration']
                    plan['codec'] = metadata['codec']
                    plan['sample_rate'] = metadata['sample_rate']
                    plan['audio_size_bytes'] = metadata['audio_size_bytes']
                    plan['video_extraction_skipped'] = metadata['video_extraction_skipped']
                    plan['source_audio_reused'] = metadata['source_audio_reused']
                    for key in ('audio_extraction_skipped', 'audio_conversion_skipped', 'audio_extraction_reason',
                                'audio_conversion_reason', 'source_audio_reused_reason'):
                        plan[key] = metadata.get(key)
                    if cloud_enabled:
                        all_jobs.extend(cloud_jobs(c, source, relative, before, metadata))
                    if c.recognition_backend == 'sarvam' or c.translation_backend == 'sarvam':
                        plan['cost'] = estimate(metadata['retained_duration'], c)
                        modes = [stage_mode(c, stage) for stage in STAGES if getattr(c, stage)
                                 and (c.recognition_backend if stage == 'transcript' else c.translation_backend) == 'sarvam']
                        jobs = math.ceil(metadata['retained_duration'] / 7200)
                        plan.update(modes=modes, jobs_per_mode=jobs, total_job_count=jobs * len(modes),
                                    eligible_for_consolidated_jobs=jobs == 1,
                                    total_estimated_cost=round(plan['cost']['estimated_cost'] * len(modes), 6))
                    audio_plans.append(plan)
            from .sarvam_provider import budget_preflight, BudgetPreflightError
            budget_report = budget_preflight(c, all_jobs, invocation=c._budget_invocation)
            over_limit = any(not plan.get('eligible_for_consolidated_jobs', True) for plan in audio_plans)
            if over_limit:
                budget_report.update(processing_permitted=False, blocked_reason='Batch duration exceeds 2 hours; automatic extra jobs are disabled')
            if args.dry_run:
                print(json.dumps({'configuration': asdict(c), 'files': [str(p) for p, _ in files],
                              'stages': [stage_settings(c, s) for s in STAGES if getattr(c, s)],
                              'audio_preprocessing': audio_plans, 'budget_preflight': budget_report,
                              'total_job_count': sum(plan.get('total_job_count', 0) for plan in audio_plans),
                              'estimated_cloud_cost_inr': round(sum(plan.get('total_estimated_cost', 0) for plan in audio_plans), 6)}, ensure_ascii=False, indent=2))
                return 0 if budget_report["processing_permitted"] else 2
            for (source, relative), plan in zip(files, audio_plans):
                directory = output_path(c, source, relative)
                with locked(directory):
                    path = directory / 'manifest.json'
                    src = identity(source)
                    manifest = json.loads(path.read_text()) if path.exists() else {'version': 1, 'source': src, 'created_at': now(), 'stages': {}}
                    if manifest.get('source') != src:
                        raise PipelineError(f'Source identity changed for {source}; choose a new output directory')
                    manifest['budget_preflight'] = budget_report
                    manifest['audio_selection_preview'] = plan
                    write_manifest(path, manifest)
            print('PREFLIGHT ' + json.dumps(budget_report, ensure_ascii=False))
            if not budget_report['processing_permitted']:
                raise BudgetPreflightError('Local Sarvam budget preflight blocked; no new paid submissions')
            # Atomically reserve all planned modes/files before starting any paid work.
            # Do not reserve for missing permission/credentials or offline execution.
            if all_jobs and c.upload_authorized and not c.offline and os.environ.get(c.sarvam_key_env):
                budget_preflight(c, all_jobs, invocation=c._budget_invocation, reserve=True)
        total = dict(completed=0, skipped=0, failed=0)
        for source, relative in files:
            try:
                counts = process(source, relative, c)
            except BudgetPreflightError as e:
                print(f'PREFLIGHT BLOCKED {source}: {e}', file=sys.stderr)
                return 2
            except Exception as e:
                print(f'ERROR {source}: {e}', file=sys.stderr)
                counts = dict(completed=0, skipped=0, failed=1)
            for key in total:
                total[key] += counts[key]
            if counts['failed'] and not c.continue_on_error:
                break
        print('Summary (stages): ' + ', '.join(f'{v} {k}' for k, v in total.items()))
        return 1 if total['failed'] else 0
    except (PipelineError, OSError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(worker_main() if sys.argv[1:] == ['--_worker'] else main())
