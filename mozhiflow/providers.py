"""Provider interfaces/adapters used by the existing stage runner."""
import copy
from typing import Protocol

from .toolkit_core import PipelineError


class RecognitionProvider(Protocol):
    def recognize(self, source, settings, context): ...


class TranslationProvider(Protocol):
    def translate(self, source, settings, context): ...


class MLXRecognition:
    def recognize(self, source, settings, context):
        return local_audio(source, settings, context)


class MLXTranslation:
    def translate(self, source, settings, context):
        return local_audio(source, settings, context)


def local_audio(source, settings, context):
    c = context['config']
    audio, prep = context['audio']() if c.preprocess_enabled else (source, None)
    result = context['inference'](audio, settings, c.offline)
    if prep is None:
        return result
    from . import pipeline as p
    from .audio_preprocess import map_segment
    result = copy.deepcopy(result)
    for segment in result.get('segments', []):
        segment['processed_start'], segment['processed_end'] = segment.get('start'), segment.get('end')
        if p.timestamp_problem(segment):
            # Invalid processed positions cannot be represented as original positions.
            segment['start'], segment['end'] = None, None
        elif not segment['text'].strip() and segment['start'] == segment['end']:
            from .audio_preprocess import map_time
            segment['start'] = segment['end'] = map_time(segment['start'], prep['mapping'])
        else:
            segment['start'], segment['end'], segment['original_spans'] = map_segment(
                segment['start'], segment['end'], prep['mapping'])
    return result


class SaarasRecognition:
    def recognize(self, source, settings, context):
        from .sarvam_provider import SarvamRecognition
        c = context['config']
        audio, prep = context['audio']()
        if 'cloud_config' in context:
            c = context['cloud_config']('codemix', prep)
        check = None
        if c.translation and c.translation_backend == 'ollama':
            from .ollama_provider import preflight
            check = lambda: preflight(c)
        def before_upload():
            if check:
                check()
            if 'cloud_budget' in context:
                context['cloud_budget'](prep)
        return SarvamRecognition().recognize(audio, prep, c, context['directory'], context['source_identity'],
                                             settings, context['record'], context['save'], before_upload=before_upload)


class SaarasTranslation:
    def translate(self, source, settings, context):
        from .sarvam_provider import SarvamRecognition
        audio, prep = context['audio']()
        c = context['cloud_config']('translate', prep)
        return SarvamRecognition().recognize(audio, prep, c, context['directory'], context['source_identity'],
            settings, context['record'], context['save'], before_upload=lambda: context['cloud_budget'](prep))


class LocalTextTranslation:
    def translate(self, source, settings, context):
        from .ollama_provider import OllamaTranslation
        return OllamaTranslation().translate(context['transcript'], context['config'], context['directory'],
                                             context['record'], context['save'])


RECOGNITION: dict[str, RecognitionProvider] = {'mlx_whisper': MLXRecognition(), 'sarvam': SaarasRecognition()}
TRANSLATION: dict[str, TranslationProvider] = {'mlx_whisper': MLXTranslation(), 'ollama': LocalTextTranslation(), 'sarvam': SaarasTranslation()}
