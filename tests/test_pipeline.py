import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import sys

from mozhiflow import pipeline as p


SOURCE = {'text': '  சரி சரி API v2 use பண்ணலாம். 42 Redis. ', 'language': 'ta',
          'segments': [{'start': 0.25, 'end': 2.5, 'text': 'சரி சரி API v2 use பண்ணலாம்.'},
                       {'start': 2.6, 'end': 4.5, 'text': '42 Redis.'}]}
ENGLISH = {'text': 'Okay okay, we can use API v2. 42 Redis.', 'language': 'ta',
           'segments': [{'start': 0.0, 'end': 4.8, 'text': 'Okay okay, we can use API v2. 42 Redis.'}]}


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.cfg = self.root / 'config.toml'

    def load(self, text='', *argv):
        self.cfg.write_text(text, encoding='utf-8')
        return p.load_config(p.parser().parse_args(['--config', str(self.cfg), *argv]))

    def test_defaults(self):
        c = self.load()
        self.assertEqual(c.model, p.LARGE)
        self.assertEqual(c.input, str(self.root / 'input'))
        self.assertTrue(c.translation and c.transcript and c.offline)

    def test_config_and_cli_precedence(self):
        c = self.load('[recognition]\nlanguage="auto"\nvocabulary=["Kafka"]\n[outputs]\ntranslation=false\n',
                      '--language', 'ta', '--vocabulary', 'API', 'Redis', '--translation', '--no-offline')
        self.assertEqual(c.language, 'ta')
        self.assertEqual(c.vocabulary, ['API', 'Redis'])
        self.assertTrue(c.translation)
        self.assertFalse(c.offline)

    def test_cli_paths_and_only(self):
        c = self.load('[outputs]\ntranscript=false\n', './meeting (1).mov', '--output-dir', './out', '--only', 'transcript')
        self.assertEqual(c.input, str(Path('./meeting (1).mov').resolve()))
        self.assertTrue(c.transcript)
        self.assertFalse(c.translation)

    def test_missing_and_malformed(self):
        for text in ('[paths', '[unknown]\nx=1', '[runtime]\nunknown=true', '[paths]\ninput=2',
                     '[outputs]\ntranscript="yes"', '[recognition]\nvocabulary=[4]',
                     '[outputs]\nformats=["vtt"]', '[outputs]\nformats=[]',
                     '[translation]\nbackend="cloud"', '[translation]\nmodel="mlx-community/whisper-large-v3-turbo"',
                     '[recognition]\nlanguage="invalid"', '[outputs]\ntranscript=false\ntranslation=false'):
            with self.subTest(text=text), self.assertRaises(p.PipelineError):
                self.load(text)
        with self.assertRaises(p.PipelineError):
            p.load_config(p.parser().parse_args(['--config', str(self.root / 'missing.toml')]))

    def test_invalid_filenames(self):
        for name in ('{stem}', '../{stage}', '{unknown}.{stage}', '{stem.__class__}.{stage}',
                     '{stage:>4}', '{stage!r}', '{stage', '{stage}/x'):
            with self.subTest(name=name), self.assertRaises(p.PipelineError):
                p.validate(p.Config(filename=name))

    def test_duplicates_and_wrong_types(self):
        for c in (p.Config(formats=['txt', 'txt']), p.Config(recursive=1), p.Config(vocabulary=['']),
                  p.Config(model='foo.en'), p.Config(filename='\n{stage}')):
            with self.assertRaises(p.PipelineError):
                p.validate(c)

    def test_translation_architecture(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text(json.dumps({'n_mels': 128, 'n_vocab': 51866, 'n_audio_layer': 32,
                                                     'n_text_layer': 32, 'n_audio_state': 1280, 'n_text_state': 1280}))
        p.validate(p.Config(translation_model=str(model)))
        (model / 'config.json').write_text('{"n_text_layer":4}')
        with self.assertRaises(p.PipelineError):
            p.validate(p.Config(translation_model=str(model)))

    def test_legacy_fast_force(self):
        c = self.load('', '--fast', '--force')
        self.assertEqual(c.model, p.TURBO)
        self.assertEqual(c.translation_model, p.LARGE)
        self.assertTrue(c.overwrite)


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / 'தமிழ் meeting (1).mov'
        self.source.write_bytes(b'original recording')
        self.c = p.Config(input=str(self.source), output_dir=str(self.root / 'out'))
        self.mock = Mock(side_effect=lambda src, settings, offline: copy.deepcopy(SOURCE if settings['stage'] == 'transcript' else ENGLISH))
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.outctx, self.errctx = redirect_stdout(self.stdout), redirect_stderr(self.stderr)
        self.outctx.__enter__()
        self.errctx.__enter__()
        self.addCleanup(self.outctx.__exit__, None, None, None)
        self.addCleanup(self.errctx.__exit__, None, None, None)

    @property
    def directory(self):
        return Path(self.c.output_dir) / self.source.name

    def run_pipeline(self):
        return p.process(self.source, Path(self.source.name), self.c, self.mock)

    def manifest(self):
        return json.loads((self.directory / 'manifest.json').read_text())

    def test_both_outputs_and_original_unchanged(self):
        before = p.identity(self.source)
        self.assertEqual(self.run_pipeline(), dict(completed=2, skipped=0, failed=0))
        self.assertEqual(before, p.identity(self.source))
        for stage, result in [('transcript', SOURCE), ('translation', ENGLISH)]:
            prefix = self.source.stem + '.' + stage
            self.assertEqual((self.directory / (prefix + '.txt')).read_text(), result['text'])
            self.assertEqual(json.loads((self.directory / (prefix + '.json')).read_text()), result)
            self.assertIn('-->', (self.directory / (prefix + '.srt')).read_text())
            record = self.manifest()['stages'][stage]
            self.assertEqual(record['status'], 'success')
            self.assertEqual(record['settings']['model'], p.LARGE)
            self.assertEqual(record['configuration']['language'], 'ta')
        self.assertIn(p.WARNING, self.stderr.getvalue())
        self.assertEqual([call.args[1]['task'] for call in self.mock.call_args_list], ['transcribe', 'translate'])

    def test_skip_and_translation_only_retry(self):
        def fail_translation(src, settings, offline):
            if settings['stage'] == 'translation':
                raise RuntimeError('out of memory')
            return SOURCE
        self.mock.side_effect = fail_translation
        self.assertEqual(self.run_pipeline(), dict(completed=1, skipped=0, failed=1))
        self.assertEqual(self.manifest()['stages']['translation']['status'], 'failed')
        self.assertIn('out of memory', self.manifest()['stages']['translation']['error'])
        self.mock.side_effect = lambda *a: ENGLISH
        self.c.transcript = False
        self.assertEqual(self.run_pipeline()['completed'], 1)
        self.assertEqual(self.mock.call_count, 3)
        self.c.transcript = True
        self.assertEqual(self.run_pipeline(), dict(completed=0, skipped=2, failed=0))
        self.assertEqual(self.mock.call_count, 3)
        self.assertEqual(self.manifest()['stages']['translation']['attempts'], 2)
        self.assertEqual(self.manifest()['stages']['transcript']['attempts'], 1)

    def test_independent_translation_after_source_failure(self):
        self.mock.side_effect = [RuntimeError('decode failure'), ENGLISH]
        self.assertEqual(self.run_pipeline(), dict(completed=1, skipped=0, failed=1))
        self.assertEqual(self.manifest()['stages']['translation']['status'], 'success')

    def test_output_can_be_enabled_independently(self):
        self.c.translation = False
        self.run_pipeline()
        self.assertEqual(self.mock.call_count, 1)
        self.assertEqual(set(self.manifest()['stages']), {'transcript'})

    def test_existing_untracked_output_is_protected(self):
        self.directory.mkdir(parents=True)
        target = self.directory / (self.source.stem + '.transcript.txt')
        target.write_text('existing')
        self.assertEqual(self.run_pipeline()['failed'], 1)
        self.assertEqual(target.read_text(), 'existing')
        self.assertEqual(self.mock.call_count, 1)  # translation still runs

    def test_changed_settings_requires_force(self):
        self.run_pipeline()
        self.c.vocabulary = ['Changed']
        self.assertEqual(self.run_pipeline()['failed'], 2)
        self.assertEqual(self.mock.call_count, 2)
        self.c.overwrite = True
        self.assertEqual(self.run_pipeline()['completed'], 2)
        self.assertEqual(self.mock.call_count, 4)

    def test_translation_settings_do_not_repeat_source(self):
        self.run_pipeline()
        self.c.translation = False
        self.c.translation_model = 'irrelevant while disabled'
        self.assertEqual(self.run_pipeline()['skipped'], 1)
        self.assertEqual(self.mock.call_count, 2)

    def test_tampered_or_missing_output_no_clobber(self):
        self.run_pipeline()
        target = self.directory / (self.source.stem + '.transcript.txt')
        target.write_text('manual corrections')
        self.assertEqual(self.run_pipeline(), dict(completed=0, skipped=1, failed=1))
        self.assertEqual(target.read_text(), 'manual corrections')
        target.unlink()
        self.assertEqual(self.run_pipeline()['failed'], 1)
        self.assertEqual(self.mock.call_count, 2)

    def test_changed_source_rejected_even_force(self):
        self.run_pipeline()
        self.source.write_bytes(b'changed')
        self.c.overwrite = True
        with self.assertRaises(p.PipelineError):
            self.run_pipeline()
        self.assertEqual(self.mock.call_count, 2)

    def test_source_changes_during_inference(self):
        def change(*a):
            self.source.write_bytes(b'changed while running')
            return SOURCE
        self.c.translation = False
        self.mock.side_effect = change
        self.assertEqual(self.run_pipeline()['failed'], 1)
        self.assertFalse(list(self.directory.glob('*.txt')))

    def test_empty_invalid_inference_and_retry(self):
        self.mock.side_effect = [{'text': '', 'segments': []}, {'text': 'x', 'segments': [{'start': 0, 'end': 1, 'text': None}]}]
        self.assertEqual(self.run_pipeline()['failed'], 2)
        self.assertFalse(list(self.directory.glob('*.txt')))
        self.mock.side_effect = [SOURCE, ENGLISH]
        self.assertEqual(self.run_pipeline()['completed'], 2)

    def test_formats_and_filename(self):
        self.c.formats = ['json']
        self.c.filename = '{stem} ({extension}) {stage}'
        self.run_pipeline()
        self.assertTrue((self.directory / f'{self.source.stem} (mov) transcript.json').is_file())
        self.assertFalse(list(self.directory.glob('*.txt')))

    def test_batch_discovery_collision_and_recursive(self):
        media = self.root / 'media'
        media.mkdir()
        (media / 'same.mov').touch()
        (media / 'same.wav').touch()
        nested = media / 'nested'
        nested.mkdir()
        (nested / 'same.MOV').touch()
        (media / 'not-media.txt').touch()
        self.c.input = str(media)
        self.assertEqual(len(p.discover(self.c)), 3)
        for source, relative in p.discover(self.c):
            p.process(source, relative, self.c, self.mock)
        self.assertEqual(len(list(Path(self.c.output_dir).rglob('manifest.json'))), 3)
        self.c.recursive = False
        self.assertEqual(len(p.discover(self.c)), 2)

    def test_output_subtree_excluded_and_invalid_input(self):
        self.c.input = str(self.root)
        self.c.output_dir = str(self.root / 'out')
        (Path(self.c.output_dir)).mkdir()
        (Path(self.c.output_dir) / 'generated.wav').touch()
        self.assertEqual(len(p.discover(self.c)), 1)
        for path in (self.root / 'missing', self.root / 'bad.txt'):
            path.touch()
            self.c.input = str(path)
            with self.assertRaises(p.PipelineError):
                p.discover(self.c)

    def test_no_continue_on_error(self):
        self.c.continue_on_error = False
        self.mock.side_effect = RuntimeError('failure')
        self.assertEqual(self.run_pipeline()['failed'], 1)
        self.assertEqual(self.mock.call_count, 1)

    def test_output_write_failure_retry(self):
        self.c.translation = False
        real_link = p.os.link
        count = 0
        def fail_second(*a):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError('disk full')
            return real_link(*a)
        with patch.object(p.os, 'link', side_effect=fail_second):
            self.assertEqual(self.run_pipeline()['failed'], 1)
        self.assertEqual(self.manifest()['stages']['transcript']['status'], 'failed')
        self.assertEqual(self.run_pipeline()['completed'], 1)
        self.assertEqual(self.mock.call_count, 1)
        self.assertEqual(self.manifest()['stages']['transcript']['attempts'], 1)

    def test_long_result_bad_segment_retains_text_and_skips_successful_translation(self):
        self.c.transcript = False
        self.run_pipeline()
        translation_before = copy.deepcopy(self.manifest()['stages']['translation'])
        self.c.transcript = True
        result = {'text': ' '.join(f'தமிழ் API {i}' for i in range(500)),
                  'segments': [{'start': i * 3.0, 'end': i * 3.0 + 2.5, 'text': f'தமிழ் API {i}'}
                               for i in range(500)]}
        result['segments'][64].update(start=227.89999999999998, end=217.89999999999998)
        self.mock.side_effect = lambda *args: copy.deepcopy(result)
        self.assertEqual(self.run_pipeline(), dict(completed=1, skipped=1, failed=0))
        prefix = self.source.stem + '.transcript'
        decoded = json.loads((self.directory / (prefix + '.json')).read_text())
        self.assertEqual(decoded['text'], result['text'])
        self.assertEqual(decoded['segments'], result['segments'])
        self.assertEqual(decoded['_timing']['invalid_segments'][0]['segment_index'], 64)
        self.assertFalse(decoded['_timing']['srt_complete'])
        self.assertEqual((self.directory / (prefix + '.txt')).read_text(), result['text'])
        srt = (self.directory / (prefix + '.srt')).read_text()
        self.assertEqual(srt.count('-->'), 499)
        self.assertNotIn('\nதமிழ் API 64\n', srt)
        self.assertIn('1 spoken cue(s) omitted', self.stderr.getvalue())
        self.assertEqual(self.manifest()['stages']['translation'], translation_before)
        self.assertTrue(p.intact(translation_before, self.directory))
        self.assertEqual(self.run_pipeline()['skipped'], 2)
        self.assertEqual(self.mock.call_count, 2)

    def test_all_invalid_times_preserve_text_and_report_empty_srt(self):
        self.c.translation = False
        result = {'text': 'API சொல்லலாம்', 'segments': [{'start': 5, 'end': 1, 'text': 'API சொல்லலாம்'}]}
        self.mock.side_effect = lambda *args: result
        self.assertEqual(self.run_pipeline()['completed'], 1)
        self.assertEqual((self.directory / (self.source.stem + '.transcript.txt')).read_text(), result['text'])
        self.assertEqual((self.directory / (self.source.stem + '.transcript.srt')).read_text(), '')
        self.assertFalse(self.manifest()['stages']['transcript']['timing']['srt_complete'])

    def test_cache_is_not_reused_after_settings_change_or_tampering(self):
        self.c.translation = False
        with patch.object(p, 'render', side_effect=OSError('disk full')):
            self.assertEqual(self.run_pipeline()['failed'], 1)
        self.c.vocabulary = ['New hint']
        with patch.object(p, 'render', side_effect=OSError('disk full')):
            self.assertEqual(self.run_pipeline()['failed'], 1)
        self.assertEqual(self.mock.call_count, 2)
        checkpoint = self.directory / '.inference/transcript.json'
        checkpoint.write_text('{}')
        self.assertEqual(self.run_pipeline()['completed'], 1)
        self.assertEqual(self.mock.call_count, 3)

    def test_dry_run_no_mutations(self):
        cfg = self.root / 'config.toml'
        cfg.write_text('')
        with patch.object(p, 'inference', side_effect=AssertionError('must not load')):
            code = p.main(['--config', str(cfg), str(self.source), '--output-dir', self.c.output_dir, '--dry-run'])
        self.assertEqual(code, 0)
        self.assertFalse(Path(self.c.output_dir).exists())

    def test_lock_and_corrupt_manifest(self):
        self.directory.mkdir(parents=True)
        with p.locked(self.directory):
            with self.assertRaises(p.PipelineError):
                self.run_pipeline()
        (self.directory / 'manifest.json').write_text('bad JSON')
        with self.assertRaises(p.PipelineError):
            self.run_pipeline()

    def test_native_worker_failure_keeps_other_stage_independent(self):
        self.c.translation = True
        with patch.object(p.subprocess, 'run', return_value=Mock(returncode=-6, stderr='Metal unavailable', stdout='')) as worker:
            counts = p.process(self.source, Path(self.source.name), self.c)
        self.assertEqual(counts['failed'], 2)
        self.assertEqual(worker.call_count, 2)
        self.assertTrue(all(r['status'] == 'failed' for r in self.manifest()['stages'].values()))
        self.assertIn('Metal unavailable', self.manifest()['stages']['transcript']['error'])

    def test_worker_valid_and_missing_result(self):
        def write_result(*args, **kwargs):
            request = json.loads(kwargs['input'])
            Path(request['result_path']).write_text(json.dumps(SOURCE))
            return Mock(returncode=0)
        with patch.object(p.subprocess, 'run', side_effect=write_result):
            result = p.isolated_inference(self.source, p.stage_settings(self.c, 'transcript'), True)
            self.assertEqual(result, SOURCE)
        with patch.object(p.subprocess, 'run', return_value=Mock(returncode=0)):
            with self.assertRaises(p.PipelineError):
                p.isolated_inference(self.source, p.stage_settings(self.c, 'transcript'), True)

    def test_timestamp_rounding(self):
        self.assertEqual(p.timestamp(3599.9999), '01:00:00,000')
        self.assertEqual(p.timestamp(0.25), '00:00:00,250')

    def test_empty_instantaneous_segments_both_stages(self):
        results = [copy.deepcopy(SOURCE), copy.deepcopy(ENGLISH)]
        empty = {'start': 0.0, 'end': 0.0, 'text': '', 'tokens': [], 'words': [],
                 'avg_logprob': float('nan')}
        for result in results:
            result['segments'].insert(0, copy.deepcopy(empty))
            result['segments'].append({'start': 30.0, 'end': 30.0, 'text': '   '})
        self.mock.side_effect = results
        self.assertEqual(self.run_pipeline()['completed'], 2)
        for stage, original in zip(p.STAGES, (SOURCE, ENGLISH)):
            prefix = self.source.stem + '.' + stage
            decoded = json.loads((self.directory / (prefix + '.json')).read_text())
            self.assertEqual(decoded['segments'][0]['start'], 0.0)
            self.assertEqual(decoded['segments'][0]['end'], 0.0)
            self.assertEqual(decoded['segments'][0]['text'], '')
            self.assertEqual(len(decoded['segments']), len(original['segments']) + 2)
            self.assertEqual((self.directory / (prefix + '.txt')).read_text(), original['text'])
            self.assertEqual((self.directory / (prefix + '.srt')).read_text(), p.render(original, 'srt'))
        self.assertEqual(self.run_pipeline()['skipped'], 2)
        self.assertEqual(self.mock.call_count, 2)

    def test_invalid_timestamps_report_index_and_values(self):
        for start, end, text in [(1, 1, 'speech'), (2, 1, ''), (-1, 0, ''),
                                 (float('nan'), 1, ''), (None, 1, '')]:
            result = copy.deepcopy(SOURCE)
            result['segments'].append({'start': start, 'end': end, 'text': text})
            with self.subTest(start=start, end=end), self.assertRaisesRegex(
                    p.PipelineError, r'timestamps at segments\[2\]: start='):
                p.validate_result(result)

    def test_only_empty_segments_do_not_count_as_speech(self):
        result = {'text': 'unexpected text', 'segments': [{'start': 0, 'end': 0, 'text': ''}]}
        with self.assertRaisesRegex(p.PipelineError, 'no nonempty timestamped speech'):
            p.prepare_result(result)

    def test_worker_preserves_upstream_empty_zero_duration_segment(self):
        result = copy.deepcopy(SOURCE)
        result['segments'].append({'start': 30.0, 'end': 30.0, 'text': ''})
        path = self.root / 'worker-empty-segment.json'
        request = {'source': str(self.source), 'settings': p.stage_settings(self.c, 'transcript'),
                   'offline': True, 'result_path': str(path)}
        with patch.object(p, 'inference', return_value=result), patch.object(sys, 'stdin', io.StringIO(json.dumps(request))):
            self.assertEqual(p.worker_main(), 0)
        self.assertEqual(json.loads(path.read_text()), result)

    def test_worker_does_not_discard_result_with_reversed_timestamps(self):
        result = copy.deepcopy(SOURCE)
        result['segments'][0].update(start=227.89999999999998, end=217.89999999999998)
        path = self.root / 'worker-reversed-segment.json'
        request = {'source': str(self.source), 'settings': p.stage_settings(self.c, 'transcript'),
                   'offline': True, 'result_path': str(path)}
        with patch.object(p, 'inference', return_value=result), patch.object(sys, 'stdin', io.StringIO(json.dumps(request))):
            self.assertEqual(p.worker_main(), 0)
        decoded = json.loads(path.read_text())
        self.assertEqual(decoded['text'], result['text'])
        self.assertEqual(decoded['segments'], result['segments'])
        self.assertFalse(decoded['_timing']['srt_complete'])

    def test_non_finite_diagnostics_both_stages(self):
        source, translation = copy.deepcopy(SOURCE), copy.deepcopy(ENGLISH)
        source['segments'][0]['avg_logprob'] = float('nan')
        translation['segments'][0]['compression_ratio'] = float('inf')
        self.mock.side_effect = [source, translation]
        self.assertEqual(self.run_pipeline()['completed'], 2)
        for stage, original, field in [('transcript', source, 'avg_logprob'),
                                       ('translation', translation, 'compression_ratio')]:
            prefix = self.source.stem + '.' + stage
            decoded = json.loads((self.directory / (prefix + '.json')).read_text(),
                                 parse_constant=lambda x: self.fail(f'Invalid JSON constant: {x}'))
            self.assertIsNone(decoded['segments'][0][field])
            self.assertEqual(decoded['text'], original['text'])
            self.assertEqual((self.directory / (prefix + '.txt')).read_text(), original['text'])
            self.assertEqual(p.render(decoded, 'srt'), p.render(original, 'srt'))
            self.assertIn('serialization', self.manifest()['stages'][stage])
        self.assertIn('$.segments[0].avg_logprob', self.stderr.getvalue())
        self.assertEqual(self.run_pipeline()['skipped'], 2)
        self.assertEqual(self.mock.call_count, 2)

    def test_non_finite_timestamps_are_preserved_as_issues(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            result = copy.deepcopy(SOURCE)
            result['segments'][0]['start'] = value
            prepared = p.prepare_result(result)
            self.assertIsNone(prepared['segments'][0]['start'])
            self.assertFalse(prepared['_timing']['srt_complete'])
            self.assertEqual(prepared['_timing']['invalid_segments'][0]['segment_index'], 0)
            json.dumps(prepared, allow_nan=False)
            with self.assertRaisesRegex(p.PipelineError, 'timestamps'):
                p.validate_result(prepared)

    def test_diagnostic_normalization_is_idempotent_and_does_not_mutate(self):
        result = copy.deepcopy(SOURCE)
        result['segments'][0]['avg_logprob'] = float('-inf')
        prepared = p.prepare_result(result)
        self.assertEqual(result['segments'][0]['avg_logprob'], float('-inf'))
        self.assertEqual(p.prepare_result(prepared), prepared)
        self.assertEqual(prepared['_serialization']['non_finite_values'],
                         [{'path': '$.segments[0].avg_logprob', 'original': '-inf'}])

    def test_worker_serializes_nan_before_transport(self):
        result = copy.deepcopy(SOURCE)
        result['segments'][0]['no_speech_prob'] = float('nan')
        path = self.root / 'worker-result.json'
        request = {'source': str(self.source), 'settings': p.stage_settings(self.c, 'transcript'),
                   'offline': True, 'result_path': str(path)}
        with patch.object(p, 'inference', return_value=result), patch.object(sys, 'stdin', io.StringIO(json.dumps(request))):
            self.assertEqual(p.worker_main(), 0)
        decoded = json.loads(path.read_text(), parse_constant=lambda x: self.fail(x))
        self.assertIsNone(decoded['segments'][0]['no_speech_prob'])
        self.assertEqual(decoded['text'], SOURCE['text'])

    def test_inference_arguments_mock_mlx(self):
        mlx = Mock()
        mlx.transcribe.return_value = SOURCE
        hub = Mock()
        hub.snapshot_download.return_value = str(self.root / 'cached')
        for stage in p.STAGES:
            with patch.dict(sys.modules, {'mlx_whisper': mlx, 'huggingface_hub': hub}), patch.object(p, 'verify_large'):
                p.inference(self.source, p.stage_settings(self.c, stage), True)
            kwargs = mlx.transcribe.call_args.kwargs
            self.assertEqual(kwargs['task'], 'transcribe' if stage == 'transcript' else 'translate')
            self.assertEqual(kwargs['language'], 'ta')
            self.assertTrue(hub.snapshot_download.call_args.kwargs['local_files_only'])
        self.c.language = 'auto'
        self.c.vocabulary = ['Kafka', 'API']
        self.c.prompt = 'Names'
        with patch.dict(sys.modules, {'mlx_whisper': mlx, 'huggingface_hub': hub}):
            p.inference(self.source, p.stage_settings(self.c, 'transcript'), True)
        self.assertIsNone(mlx.transcribe.call_args.kwargs['language'])
        self.assertEqual(mlx.transcribe.call_args.kwargs['initial_prompt'], 'Names\nKafka, API')


if __name__ == '__main__':
    unittest.main()
