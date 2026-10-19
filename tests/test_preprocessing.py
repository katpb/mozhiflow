import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from mozhiflow import audio_preprocess as a
from mozhiflow.toolkit_core import PipelineError
from types import SimpleNamespace


def config():
    return SimpleNamespace(preprocess_enabled=True, normalize_audio=True, vad_backend='none', vad_threshold=0.35,
                           vad_min_silence=5.0, vad_padding=0.75, vad_min_speech_ms=250, vad_max_intervals=200)


class PreprocessingTests(unittest.TestCase):
    def test_conservative_padding_and_merge(self):
        result = a.retained_intervals(1000, [{'start': 100, 'end': 200}, {'start': 220, 'end': 300},
                                            {'start': 700, 'end': 800}], 20, 100)
        self.assertEqual(result, [(0, 320), (680, 820)])
        self.assertEqual(a.retained_intervals(1000, [], 20, 100), [(0, 1000)])

    def test_no_removal_for_short_gaps_or_edge_silence(self):
        self.assertEqual(a.retained_intervals(1000, [{'start': 50, 'end': 950}], 20, 100), [(0, 1000)])

    def test_invalid_vad_intervals(self):
        for speech in ([{'start': 2, 'end': 1}], [{'start': -1, 'end': 10}], [{'start': 0, 'end': 1001}]):
            with self.assertRaises(PipelineError):
                a.retained_intervals(1000, speech, 20, 100)

    def test_mapping_boundaries_and_crossing_gap(self):
        mapping = a.build_mapping([(0, a.RATE * 10), (a.RATE * 20, a.RATE * 30)], 0.5)
        self.assertEqual(a.map_time(10, mapping), 20.5)
        self.assertEqual(a.map_time(10, mapping, end=True), 10.5)
        self.assertEqual(a.map_time(20, mapping), 30.5)
        start, end, spans = a.map_segment(9, 11, mapping)
        self.assertEqual((start, end), (9.5, 21.5))
        self.assertEqual(spans, [{'start': 9.5, 'end': 10.5}, {'start': 20.5, 'end': 21.5}])
        for value in (-1, 25, float('nan')):
            with self.assertRaises(PipelineError):
                a.map_time(value, mapping)

    def test_real_ffmpeg_no_vad_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'தமிழ் (1).wav'
            with wave.open(str(source), 'wb') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(a.RATE)
                f.writeframes(b'\0\0' * a.RATE)
            before = source.read_bytes()
            output, record = a.prepare(source, root / 'out', config(), {'sha256': 'test'})
            self.assertEqual(record['retained_duration'], 1.0)
            self.assertEqual(record['removed_duration'], 0.0)
            self.assertEqual(source.read_bytes(), before)
            with patch.object(a, 'command', side_effect=AssertionError('cache should be reused')):
                self.assertEqual(a.prepare(source, root / 'out', config(), {'sha256': 'test'})[1], record)

    def test_mock_silero_no_speech_conservative_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.wav'
            with wave.open(str(source), 'wb') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(a.RATE)
                f.writeframes(b'\0\0' * a.RATE)
            c = config(); c.vad_backend = 'silero'
            with patch.object(a, 'detect_speech', return_value=([], 'test-version')):
                _, record = a.prepare(source, Path(tmp) / 'out', c, {})
            self.assertTrue(record['no_speech_fallback'])
            self.assertEqual(record['vad_model_version'], 'test-version')
            self.assertEqual(record['removed_duration'], 0)

    def test_mock_silero_removes_only_long_gap_interiors_and_copies_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.wav'
            samples = b'\x01\0' * a.RATE * 20
            with wave.open(str(source), 'wb') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(a.RATE); f.writeframes(samples)
            c = config(); c.vad_backend = 'silero'; c.normalize_audio = False
            speech = [{'start': 2*a.RATE, 'end': 4*a.RATE}, {'start': 12*a.RATE, 'end': 14*a.RATE}]
            with patch.object(a, 'detect_speech', return_value=(speech, 'test-version')):
                output, record = a.prepare(source, Path(tmp)/'out', c, {})
            self.assertEqual(record['retained_duration'], 8.25)
            self.assertEqual(record['removed_duration'], 11.75)
            with wave.open(str(output), 'rb') as f:
                self.assertEqual(f.getnframes(), int(8.25*a.RATE))
                self.assertEqual(f.readframes(f.getnframes()), b'\x01\0' * int(8.25*a.RATE))
            self.assertEqual(a.map_time(4.75, record['mapping']), 11.25)
