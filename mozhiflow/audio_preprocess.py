"""Local FFmpeg extraction, conservative VAD and sample-accurate time mapping."""
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import wave
from fractions import Fraction

from .toolkit_core import PipelineError, atomic_json, sha256

RATE = 16000


def settings(c):
    result = {'enabled': c.preprocess_enabled, 'normalize': c.normalize_audio,
            'backend': c.vad_backend, 'threshold': c.vad_threshold,
            'min_silence_seconds': c.vad_min_silence, 'padding_seconds': c.vad_padding,
            'min_speech_ms': c.vad_min_speech_ms, 'max_intervals': c.vad_max_intervals,
            'sample_rate': getattr(c, 'audio_sample_rate', RATE), 'version': 1}
    # Keep old WAV/16 kHz preparation fingerprints reusable.
    if getattr(c, 'audio_codec', 'wav') != 'wav':
        result['codec'] = c.audio_codec
    return result


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout
    except (OSError, subprocess.CalledProcessError) as e:
        # No authentication or signed URLs are ever passed to FFmpeg.
        raise PipelineError(f'Local audio preprocessing failed: {e}') from e


def probe(source):
    data = json.loads(command(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                               '-show_entries', 'format=duration,start_time:stream=start_time', '-of', 'json', str(source)]))
    if not data.get('streams'):
        raise PipelineError('Recording has no audio stream')
    duration = float(data['format']['duration'])
    start = float(data['format'].get('start_time', 0))
    offset = float(data['streams'][0].get('start_time', start)) - start
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(offset) or offset < -1 / RATE:
        raise PipelineError('Invalid source duration or audio timeline offset')
    return duration, max(0, offset)


def media_info(source):
    # Scan packets as well as headers: demux errors/empty streams must fail locally.
    args = ['ffprobe', '-v', 'error', '-count_packets', '-show_error', '-show_entries',
        'format=format_name:stream=codec_type,codec_name,sample_rate,channels,duration,duration_ts,time_base,nb_read_packets:stream_disposition=attached_pic',
        '-of', 'json', str(source)]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode or result.stderr.strip():
        raise PipelineError('ffprobe media integrity inspection failed; source was not modified')
    data = json.loads(result.stdout)
    streams = data.get('streams', [])
    audios = [stream for stream in streams if stream.get('codec_type') == 'audio']
    if not audios or int(audios[0].get('nb_read_packets', 0)) <= 0:
        raise PipelineError('Recording has no nonempty audio stream')
    audio = audios[0]
    if int(audio.get('sample_rate', 0)) <= 0 or int(audio.get('channels', 0)) <= 0:
        raise PipelineError('Invalid audio sample rate or channel count')
    audio['container'] = data.get('format', {}).get('format_name', '')
    audio['audio_stream_count'] = len(audios)
    audio_only = not any(stream.get('codec_type') == 'video'
                         and not stream.get('disposition', {}).get('attached_pic') for stream in streams)
    return audio_only, audio


def compatible_format(stream):
    """Conservative codec/container pairs from Sarvam's supported-format documentation.

    Return a canonical upload extension and MIME, independently of the source suffix.
    Unknown combinations are decoded to the configured WAV/FLAC intermediate.
    """
    containers = set(stream.get('container', '').split(','))
    codec = stream.get('codec_name', '')
    pairs = [('m4a', 'audio/mp4', {'mov', 'mp4', 'm4a'}, {'aac'}),
             ('aac', 'audio/aac', {'aac'}, {'aac'}),
             ('mp3', 'audio/mpeg', {'mp3'}, {'mp3'}),
             ('flac', 'audio/flac', {'flac'}, {'flac'}),
             ('ogg', 'audio/ogg', {'ogg'}, {'opus', 'vorbis'}),
             ('wav', 'audio/wav', {'wav'}, {'pcm_s16le', 'pcm_s24le', 'pcm_s32le', 'pcm_f32le', 'pcm_u8'}),
             ('aiff', 'audio/aiff', {'aiff'}, {'pcm_s16be', 'pcm_s24be', 'pcm_s32be'})]
    for extension, mime, formats, codecs in pairs:
        if containers & formats and codec in codecs and stream.get('audio_stream_count', 1) == 1:
            return extension, mime
    return None


def decisions(audio_only, reused, reason):
    return {'audio_selection_version': 2, 'audio_extraction_skipped': audio_only,
            'audio_conversion_skipped': reused, 'source_audio_reused': reused,
            'audio_extraction_reason': 'Audio-only source; video extraction unnecessary' if audio_only else 'Video stream present; extract first audio stream',
            'audio_conversion_reason': reason, 'source_audio_reused_reason': reason,
            'video_extraction_skipped': audio_only}


def detect_speech(wav_path, c):
    try:
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps
    except ImportError as e:
        raise PipelineError('Silero VAD is not installed; install requirements-hybrid.txt or set preprocessing.vad_backend="none"') from e
    with wave.open(str(wav_path), 'rb') as f:
        raw = bytearray(f.readframes(f.getnframes()))
    samples = torch.frombuffer(raw, dtype=torch.int16).to(torch.float32) / 32768.0
    # Model weights are included in the installed silero-vad package; no torch.hub download.
    model = load_silero_vad(onnx=False)
    intervals = get_speech_timestamps(samples, model, sampling_rate=getattr(c, 'audio_sample_rate', RATE), threshold=c.vad_threshold,
                                     min_speech_duration_ms=c.vad_min_speech_ms,
                                     min_silence_duration_ms=100, speech_pad_ms=0, return_seconds=False)
    return intervals, importlib.metadata.version('silero-vad')


def retained_intervals(total, speech, padding, min_silence):
    """Remove only the interior of long gaps. Short gaps remain continuous."""
    if total <= 0:
        raise PipelineError('Extracted audio is empty')
    if not speech:
        # Conservative policy: uncertain/no detected speech never silently deletes the recording.
        return [(0, total)]
    padded = []
    for item in speech:
        a, b = item['start'], item['end']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= total:
            raise PipelineError('VAD returned invalid speech intervals')
        a, b = max(0, a - padding), min(total, b + padding)
        if padded and a < padded[-1][0]:
            raise PipelineError('VAD intervals are not ordered')
        if padded and a - padded[-1][1] < min_silence:
            padded[-1] = (padded[-1][0], max(b, padded[-1][1]))
        else:
            padded.append((a, b))
    # Keep short leading/trailing non-speech too.
    if padded[0][0] < min_silence:
        padded[0] = (0, padded[0][1])
    if total - padded[-1][1] < min_silence:
        padded[-1] = (padded[-1][0], total)
    return padded


def build_mapping(intervals, origin_offset=0.0, sample_rate=RATE):
    result, cursor = [], 0
    for a, b in intervals:
        result.append({'processed_start_sample': cursor, 'processed_end_sample': cursor + b - a,
                       'original_start_sample': a, 'original_end_sample': b,
                       'original_start': a / sample_rate + origin_offset, 'original_end': b / sample_rate + origin_offset})
        if sample_rate != RATE:
            result[-1]['sample_rate'] = sample_rate
        cursor += b - a
    return result


def map_time(seconds, mapping, *, end=False):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise PipelineError(f'Invalid provider timestamp: {seconds!r}')
    rate = mapping[0].get('sample_rate', RATE)
    sample = seconds * rate
    maximum = mapping[-1]['processed_end_sample']
    if sample > maximum + 1:  # tolerate at most one sample of provider rounding
        raise PipelineError('Provider timestamp exceeds processed audio duration')
    sample = min(sample, maximum)
    for index, part in enumerate(mapping):
        a, b = part['processed_start_sample'], part['processed_end_sample']
        # At a removed-gap boundary, starts belong to the next interval and ends to the previous.
        if a <= sample < b or (end and a < sample <= b) or (index == len(mapping) - 1 and sample == b):
            return part['original_start'] + (sample - a) / rate
    raise PipelineError('Provider timestamp cannot be mapped')


def map_segment(start, end, mapping):
    if end <= start:
        raise PipelineError('Provider returned non-positive chunk duration')
    a, b = map_time(start, mapping), map_time(end, mapping, end=True)
    rate = mapping[0].get('sample_rate', RATE)
    spans = []
    for part in mapping:
        left = max(start * rate, part['processed_start_sample'])
        right = min(end * rate, part['processed_end_sample'])
        if left < right:
            spans.append({'start': part['original_start'] + (left - part['processed_start_sample']) / rate,
                          'end': part['original_start'] + (right - part['processed_start_sample']) / rate})
    return a, b, spans


def prepare(source, directory, c, source_identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    codec = getattr(c, 'audio_codec', 'wav')
    rate = getattr(c, 'audio_sample_rate', RATE)
    output = directory / f'processed.{codec}'
    if any((directory / name).is_symlink() for name in ('normalized.wav', 'processed.wav', 'processed.flac', 'preprocessing.json')):
        raise PipelineError('Refusing symlinks in preprocessing artifacts')
    record_path = directory / 'preprocessing.json'
    config = settings(c)
    if record_path.exists():
        old = json.loads(record_path.read_text())
        cached_audio = Path(source) if old.get('source_audio_reused') else output
        if (old.get('source') == source_identity and old.get('configuration') == config
                and cached_audio.is_file() and old.get('audio_sha256') == sha256(cached_audio)):
            active = c.normalize_audio or c.vad_backend != 'none'
            # Preserve the exact payload of historical/ambiguous paid work across upgrades.
            paid = any(json.loads(p.read_text()).get('audio_sha256') == old.get('audio_sha256')
                       and json.loads(p.read_text()).get('phase') not in {'new', 'imported'}
                       for p in directory.parents[1].glob('.providers/sarvam/*/remote.json'))
            if not old.get('audio_selection_version') and not active and not paid:
                audio_only, stream = media_info(source)
                if audio_only and compatible_format(stream):
                    return prepare(source, directory / 'native-v2', c, source_identity)
            if not old.get('audio_selection_version'):
                old.update(decisions(old.get('input_kind') == 'audio', old.get('source_audio_reused', False),
                                     'Retaining cached payload for existing paid-job recovery' if paid else 'Cached processed audio'))
            old.update(audio_size_bytes=cached_audio.stat().st_size)
            scratch = directory / 'normalized.wav'
            if (not getattr(c, 'retain_intermediates', False) and scratch.is_file()
                    and scratch.resolve() != Path(source).resolve()):
                scratch.unlink()
            return cached_audio, old
    original_duration, offset = probe(source)
    audio_only, stream = media_info(source)
    supported = compatible_format(stream)
    can_reuse = audio_only and not c.normalize_audio and c.vad_backend == 'none' and supported is not None
    reason = ('Supported codec/container and no active normalization or VAD' if can_reuse else
              'Video requires audio extraction' if not audio_only else
              'Normalization or VAD enabled' if c.normalize_audio or c.vad_backend != 'none' else
              'Unsupported codec/container or multiple audio streams; convert first audio stream')
    if can_reuse:
        rate = int(stream['sample_rate'])
        codec = stream['codec_name']
        duration = (float(int(stream['duration_ts']) * Fraction(stream['time_base']))
                    if stream.get('duration_ts') is not None and stream.get('time_base')
                    else float(stream.get('duration', original_duration - offset)))
        total = round(duration * rate)
        if total <= 0:
            raise PipelineError('Input audio is empty')
        record = {'status': 'success', 'source': source_identity, 'configuration': config,
                  'input_kind': 'audio', 'video_extraction_skipped': True, 'source_audio_reused': True,
                  'original_duration': original_duration, 'audio_duration': total / rate,
                  'retained_duration': total / rate, 'removed_duration': 0.0, 'removed_intervals': [],
                  'predicted_savings_percent': 0.0, 'audio_origin_offset': offset,
                  'mapping': build_mapping([(0, total)], offset, rate), 'codec': codec, 'sample_rate': rate,
                  'audio_size_bytes': Path(source).stat().st_size, 'audio_sha256': sha256(source),
                  'detected_speech_intervals': None, 'no_speech_fallback': False,
                  'vad_model_version': None, 'ffmpeg_version': command(['ffmpeg', '-version']).splitlines()[0]}
        record.update(decisions(True, True, reason), input_stream=stream,
                      upload_extension=supported[0], upload_mime=supported[1],
                      integrity_check='ffprobe full packet scan; not a full decoder validation')
        atomic_json(record_path, record)
        return Path(source), record
    normalized = directory / 'normalized.wav'
    filters = 'loudnorm=I=-20:TP=-2:LRA=11,' if c.normalize_audio else ''
    command(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source), '-map', '0:a:0',
             '-vn', '-af', filters + 'asetpts=PTS-STARTPTS', '-ac', '1', '-ar', str(rate),
             '-c:a', 'pcm_s16le', str(normalized)])
    with wave.open(str(normalized), 'rb') as f:
        total, params = f.getnframes(), f.getparams()
    speech, version = (detect_speech(normalized, c) if c.vad_backend == 'silero' else (None, None))
    intervals = (retained_intervals(total, speech, round(c.vad_padding * rate), round(c.vad_min_silence * rate))
                 if speech is not None else [(0, total)])
    if len(intervals) > c.vad_max_intervals:
        raise PipelineError('VAD would excessively fragment speech; increase minimum silence or disable VAD')
    consolidated = directory / 'processed.wav'
    with wave.open(str(normalized), 'rb') as inp, wave.open(str(consolidated), 'wb') as out:
        out.setparams(params)
        for a, b in intervals:
            inp.setpos(a)
            remaining = b - a
            while remaining:
                n = min(remaining, rate * 60)
                out.writeframesraw(inp.readframes(n))
                remaining -= n
    if codec == 'flac':
        command(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(consolidated),
                 '-c:a', 'flac', '-compression_level', '8', str(output)])
        consolidated.unlink()
    retained = sum(b - a for a, b in intervals) / rate
    removed, cursor = [], 0
    for a, b in intervals:
        if cursor < a:
            removed.append({'start': cursor / rate + offset, 'end': a / rate + offset})
        cursor = b
    if cursor < total:
        removed.append({'start': cursor / rate + offset, 'end': total / rate + offset})
    record = {'status': 'success', 'source': source_identity, 'configuration': config,
              'input_kind': 'audio' if audio_only else 'video', 'video_extraction_skipped': audio_only,
              'source_audio_reused': False,
              'vad_model_version': version, 'ffmpeg_version': command(['ffmpeg', '-version']).splitlines()[0],
              'original_duration': original_duration, 'audio_duration': total / rate,
              'retained_duration': retained, 'removed_duration': total / rate - retained,
              'predicted_savings_percent': 100 * (1 - retained / (total / rate)),
              'audio_origin_offset': offset, 'mapping': build_mapping(intervals, offset, rate),
              'removed_intervals': removed, 'codec': codec, 'sample_rate': rate,
              'audio_size_bytes': output.stat().st_size,
              'detected_speech_intervals': speech, 'no_speech_fallback': speech == [],
              'audio_sha256': sha256(output)}
    record.update(decisions(audio_only, False, reason), input_stream=stream,
                  upload_extension=codec, upload_mime='audio/' + codec,
                  integrity_check='ffprobe packet scan and FFmpeg decode')
    atomic_json(record_path, record)
    if not getattr(c, 'retain_intermediates', False):
        normalized.unlink()
    return output, record
