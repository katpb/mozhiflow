import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import wave

from mozhiflow import pipeline as p
from mozhiflow import sarvam_provider as s
from mozhiflow import ollama_provider as o
from mozhiflow.toolkit_core import PipelineError


class HybridIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.source = self.root / 'தமிழ் meeting (1).wav'
        with wave.open(str(self.source), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(b'\0\0' * 16000 * 2)
        self.c = p.Config(input=str(self.source), output_dir=str(self.root / 'out'), recognition_backend='sarvam',
                          model='saaras:v4', translation_backend='ollama', translation_model='gemma3:12b',
                          preprocess_enabled=True, offline=False, sarvam_budget=1.0)
        p.validate(self.c)
        self.log = io.StringIO()
        self.out = redirect_stdout(self.log); self.err = redirect_stderr(self.log)
        self.out.__enter__(); self.err.__enter__()
        self.addCleanup(self.out.__exit__, None,None,None); self.addCleanup(self.err.__exit__,None,None,None)
        self.sarvam_calls = 0; self.translation_calls = 0
        def recognize(provider, audio, prep, c, directory, src, settings, record, save, **kwargs):
            self.sarvam_calls += 1
            record['provider'] = {'backend': 'sarvam', 'remote_job_id': 'job_test', 'model': 'saaras:v4'}
            save()
            return s.normalize_response({'transcript': 'API 42 வேண்டாம்', 'timestamps': {'chunks': ['API 42 வேண்டாம்'],
                'start_time_seconds': [0.2], 'end_time_seconds': [1.8]}}, prep['mapping'])
        self.recognize = recognize
        self.client = Mock()
        def request(path, payload=None):
            if path == '/api/tags': return {'models': [{'name': self.c.translation_model, 'digest': 'local-digest'}]}
            self.translation_calls += 1
            return {'done': True, 'response': json.dumps({'segments': [{'id':'s000000','text':'Do not use API 42.'}]})}
        self.client.request.side_effect = request

    @property
    def directory(self): return Path(self.c.output_dir) / self.source.name

    def run_pipeline(self):
        with patch.object(s.SarvamRecognition, 'recognize', self.recognize), patch.object(o, 'OllamaClient', return_value=self.client):
            return p.process(self.source, Path(self.source.name), self.c)

    def test_end_to_end_outputs_retry_and_selective_translation_change(self):
        before = p.identity(self.source)
        self.assertEqual(self.run_pipeline(), {'completed':2,'skipped':0,'failed':0})
        transcript = json.loads((self.directory / (self.source.stem + '.transcript.json')).read_text())
        english = json.loads((self.directory / (self.source.stem + '.translation.json')).read_text())
        for key in ('id','start','end','original_spans'):
            self.assertEqual(transcript['segments'][0][key], english['segments'][0][key])
        for stage in p.STAGES:
            for fmt in ('txt','srt','json'):
                self.assertTrue((self.directory / f'{self.source.stem}.{stage}.{fmt}').stat().st_size)
        self.assertEqual(self.run_pipeline()['skipped'], 2)
        self.assertEqual((self.sarvam_calls,self.translation_calls),(1,1))
        self.c.transcript=False; self.c.translation_prompt='Preserve conversational tone.'; self.c.overwrite=True
        self.assertEqual(self.run_pipeline()['completed'],1)
        self.assertEqual(self.sarvam_calls,1)
        self.assertEqual(p.identity(self.source),before)
        m=json.loads((self.directory/'manifest.json').read_text())
        self.assertEqual(m['stages']['transcript']['provider']['remote_job_id'],'job_test')
        self.assertEqual(m['preprocessing']['status'],'success')
        self.assertEqual(m['stages']['translation']['provider']['model_digest'],'local-digest')

    def test_translation_failure_retains_source_and_only_translation_retry(self):
        original = self.client.request.side_effect
        self.client.request.side_effect=lambda path,payload=None: original(path,payload) if path=='/api/tags' else {'done':True,'response':'bad'}
        self.assertEqual(self.run_pipeline(),{'completed':1,'skipped':0,'failed':1})
        self.assertTrue((self.directory / (self.source.stem+'.transcript.txt')).is_file())
        self.client.request.side_effect=original; self.c.transcript=False
        self.assertEqual(self.run_pipeline()['completed'],1)
        self.assertEqual(self.sarvam_calls,1)

    def test_translation_only_never_invokes_recognition_when_source_missing(self):
        self.c.transcript=False
        self.assertEqual(self.run_pipeline()['failed'],1)
        self.assertEqual(self.sarvam_calls,0)

    def test_dry_run_has_durations_and_cost_without_remote_calls(self):
        cfg=self.root/'hybrid.toml'
        cfg.write_text('[recognition]\nbackend="sarvam"\nmodel="saaras:v4"\n[translation]\nbackend="ollama"\n[preprocessing]\nenabled=true\n[runtime]\noffline=false\n')
        with patch.object(s.Transport,'api',side_effect=AssertionError('must stay local')):
            self.assertEqual(p.main(['--config',str(cfg),str(self.source),'--dry-run']),2)
        plan=json.loads(self.log.getvalue())
        self.assertEqual(plan['configuration']['translation_model'],'gemma3:12b')
        self.assertEqual(plan['audio_preprocessing'][0]['retained_duration'],2)
        self.assertGreater(plan['audio_preprocessing'][0]['cost']['estimated_cost'],0)
        self.assertFalse((self.root/'out').exists())

    def test_new_configuration_validation_and_cli_consent(self):
        cfg=self.root/'hybrid.toml'
        cfg.write_text('[recognition]\nbackend="sarvam"\nmodel="saaras:v4"\n[translation]\nbackend="ollama"\n[sarvam]\nbudget_inr=1\n[runtime]\noffline=false\n')
        c=p.load_config(p.parser().parse_args(['--config',str(cfg),'--allow-upload','--sarvam-budget','2']))
        self.assertTrue(c.upload_authorized); self.assertEqual(c.sarvam_budget,2.0)
        cfg.write_text('[sarvam]\nallow_upload=true\n')
        with self.assertRaises(PipelineError): p.load_config(p.parser().parse_args(['--config',str(cfg)]))
        for changes in ({'vad_threshold':float('nan')},{'vad_padding':-1.0},{'sarvam_budget':-1},
                        {'model':'saaras:v3'},{'ollama_endpoint':'https://external.example'}, {'vocabulary':['x'*65]}):
            c=copy.deepcopy(self.c)
            for key,value in changes.items(): setattr(c,key,value)
            with self.assertRaises(PipelineError): p.validate(c)
