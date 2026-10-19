"""Hermetic tests: actual local FFmpeg, fake Batch network, no live provider calls."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from mozhiflow import audio_preprocess as audio
from mozhiflow import pipeline as p
from mozhiflow import sarvam_provider as s
from mozhiflow.toolkit_core import PipelineError, sha256


class BatchDouble:
    def __init__(self):
        self.jobs = {}; self.calls = []; self.uploads = []; self.failed = set(); self.pending = set()
        self.no_timestamps = set(); self.interrupt_create = None

    def api(self, method, path='', payload=None):
        self.calls.append((method,path,copy.deepcopy(payload)))
        if method == 'POST' and not path:
            mode = payload['job_parameters']['mode']
            job_id = f'job_{len(self.jobs)}'
            self.jobs[job_id] = mode
            if mode == self.interrupt_create:
                raise KeyboardInterrupt('connection interrupted after server received create')
            return {'job_id':job_id}
        if path == '/upload-files':
            name=payload['files'][0]
            return {'storage_container_type':'Azure_V1','upload_urls':{name:{'file_url':'https://storage/'+payload['job_id']}}}
        if path.endswith('/start'): return {'job_state':'Running'}
        if path.endswith('/status'):
            job=path.split('/')[1]; mode=self.jobs[job]
            if mode in self.failed: return {'job_state':'Failed'}
            if mode in self.pending: return {'job_state':'Running'}
            return {'job_state':'Completed','failed_files_count':0,'job_details':[{'state':'Success','outputs':[{'file_name':'0.json'}]}]}
        if path == '/download-files':
            return {'download_urls':{'0.json':{'file_url':'https://result/'+payload['job_id']}}}
        raise AssertionError((method,path,payload))

    def upload(self,url,path,storage): self.uploads.append((str(path),sha256(path)))

    def download(self,url):
        mode=self.jobs[url.split('/')[-1]]
        result={'transcript':'தமிழ் API 42' if mode=='codemix' else 'Use API 42.', 'language_code':'ta-IN'}
        if mode not in self.no_timestamps:
            result['timestamps']={'chunks':[result['transcript']], 'start_time_seconds':[0.1], 'end_time_seconds':[1.8]}
        return (json.dumps(result,ensure_ascii=False)+'\n').encode()


class DualSarvamTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve(); self.source=self.root/'தமிழ் meeting (1).wav'
        with wave.open(str(self.source),'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(b'\x01\0'*32000)
        self.c=p.Config(input=str(self.source),output_dir=str(self.root/'out'),model='saaras:v4',
            recognition_backend='sarvam',translation_backend='sarvam',translation_model='saaras:v4',
            preprocess_enabled=True,normalize_audio=False,audio_codec='flac',sarvam_budget=10.0,
            upload_authorized=True,offline=False,filename='{stem}.{mode}',output_directory='{stem}',
            sarvam_timestamp_policy='flag',sarvam_poll_timeout=0.01,sarvam_poll_interval=0.1)
        p.validate(self.c)
        self.fake=BatchDouble(); self.log=io.StringIO()
        for ctx in (redirect_stdout(self.log),redirect_stderr(self.log)):
            ctx.__enter__(); self.addCleanup(ctx.__exit__,None,None,None)
        self.env=patch.dict(os.environ,{'SARVAM_API_KEY':'unit-test-secret'}); self.env.start(); self.addCleanup(self.env.stop)

    @property
    def directory(self): return p.output_path(self.c,self.source,Path(self.source.name))
    def manifest(self): return json.loads((self.directory/'manifest.json').read_text())
    def run_pipeline(self):
        with patch.object(s,'Transport',return_value=self.fake):
            return p.process(self.source,Path(self.source.name),self.c)

    def test_both_modes_share_one_audio_payload_and_skip(self):
        before=p.identity(self.source)
        with patch('mozhiflow.ollama_provider.preflight',side_effect=AssertionError('Ollama must not be required')):
            self.assertEqual(self.run_pipeline(),{'completed':2,'skipped':0,'failed':0})
        self.assertEqual(list(self.fake.jobs.values()),['codemix','translate'])
        self.assertEqual(len(set(self.fake.uploads)),1)
        self.assertEqual(self.fake.uploads[0][0],str(self.source))
        m=self.manifest()
        self.assertEqual(m['preprocessing']['codec'],'pcm_s16le')
        ids=[m['stages'][stage]['provider']['remote_job_id'] for stage in p.STAGES]
        self.assertEqual(len(set(ids)),2)
        for stage,mode in [('transcript','codemix'),('translation','translate')]:
            record=m['stages'][stage]
            self.assertEqual(record['provider']['mode'],mode)
            self.assertEqual(record['provider']['remote_status'],'Completed')
            for fmt in ('txt','srt','json'):
                self.assertTrue((self.directory/f'{self.source.stem}.{mode}.{fmt}').stat().st_size)
            self.assertTrue((self.directory/record['provider']['raw_response']).is_file())
        self.assertEqual(p.identity(self.source),before)
        self.assertNotIn('unit-test-secret',json.dumps(m))
        self.c.upload_authorized=False
        self.assertEqual(self.run_pipeline(),{'completed':0,'skipped':2,'failed':0})
        self.assertEqual(len(self.fake.jobs),2)

    def test_codemixture_failure_does_not_block_translation_or_buy_replacement(self):
        self.fake.failed.add('codemix')
        self.assertEqual(self.run_pipeline(),{'completed':1,'skipped':0,'failed':1})
        self.assertEqual(self.manifest()['stages']['translation']['status'],'success')
        self.assertEqual(self.run_pipeline(),{'completed':0,'skipped':1,'failed':1})
        self.assertEqual(len(self.fake.jobs),2)

    def test_pending_translation_resumes_without_reuploading(self):
        self.fake.pending.add('translate')
        self.assertEqual(self.run_pipeline()['failed'],1)
        self.c.upload_authorized=False; self.fake.pending.clear()
        self.assertEqual(self.run_pipeline(),{'completed':1,'skipped':1,'failed':0})
        self.assertEqual(len(self.fake.jobs),2); self.assertEqual(len(self.fake.uploads),2)

    def test_saved_consent_survives_output_failure_before_second_job(self):
        self.c.continue_on_error=False
        with patch.object(p,'render',side_effect=OSError('disk full')):
            self.assertEqual(self.run_pipeline()['failed'],1)
        self.assertEqual(len(self.fake.jobs),1)
        self.c.upload_authorized=False
        self.assertEqual(self.run_pipeline()['completed'],2)
        self.assertEqual(len(self.fake.jobs),2)

    def test_interrupted_create_reconciles_only_translation(self):
        self.fake.interrupt_create='translate'
        with self.assertRaises(KeyboardInterrupt): self.run_pipeline()
        self.c.upload_authorized=False; self.fake.interrupt_create=None
        self.assertEqual(self.run_pipeline()['failed'],1)
        self.assertEqual(len(self.fake.jobs),2)
        self.c.sarvam_translation_job_id='job_1'
        self.assertEqual(self.run_pipeline(),{'completed':1,'skipped':1,'failed':0})
        self.assertEqual(len(self.fake.jobs),2)

    def test_combined_budget_checked_before_first_paid_job(self):
        self.c.sarvam_budget=1.0 # Each estimate fits; BOTH estimates do not.
        with self.assertRaises(s.BudgetPreflightError):
            self.run_pipeline()
        self.assertEqual(len(self.fake.jobs),0)
        self.assertFalse((Path(self.c.output_dir)/'.sarvam-budget.json').exists())

    def test_missing_timestamps_preserve_text_without_fake_srt(self):
        self.fake.no_timestamps.add('translate')
        self.assertEqual(self.run_pipeline()['completed'],2)
        self.assertEqual((self.directory/f'{self.source.stem}.translate.txt').read_text(),'Use API 42.')
        self.assertEqual((self.directory/f'{self.source.stem}.translate.srt').read_text(),'')
        result=json.loads((self.directory/f'{self.source.stem}.translate.json').read_text())
        self.assertIsNone(result['segments'][0]['start'])
        self.assertFalse(result['_timing']['srt_complete'])
        self.assertIn('omitted from SRT',self.log.getvalue())

    def test_duration_above_limit_does_not_split_or_create_jobs(self):
        original=audio.prepare
        def long(*args,**kwargs):
            path,meta=original(*args,**kwargs); meta['retained_duration']=7201.0
            return path,meta
        with patch.object(audio,'prepare',side_effect=long):
            self.assertEqual(self.run_pipeline()['failed'],2)
        self.assertFalse(self.fake.jobs)
        self.assertIn('4 total selected jobs',self.log.getvalue())

    def test_environment_key_name_is_configurable(self):
        self.c.sarvam_key_env='MY_SARVAM_KEY'
        with patch.dict(os.environ,{'MY_SARVAM_KEY':'another-private-key','PATH':os.environ.get('PATH','')},clear=True):
            self.assertEqual(self.run_pipeline()['completed'],2)
        self.assertNotIn('another-private-key',json.dumps(self.manifest()))

    def test_translation_only_needs_no_codemixture_checkpoint(self):
        self.c.transcript=False
        self.assertEqual(self.run_pipeline()['completed'],1)
        self.assertEqual(list(self.fake.jobs.values()),['translate'])

    def test_definitive_credit_rejection_can_retry_without_unknown_create(self):
        original=self.fake.api; count=[0]
        def reject(method,path='',payload=None):
            if method=='POST' and not path and count[0]==0:
                count[0]+=1; raise s.SarvamHTTPError(402)
            return original(method,path,payload)
        self.fake.api=reject; self.c.translation=False
        self.assertEqual(self.run_pipeline()['failed'],1)
        self.assertEqual(self.run_pipeline()['completed'],1)
        self.assertEqual(len(self.fake.jobs),1)

    def test_two_hour_boundary_accepts_exactly_two_jobs(self):
        original=audio.prepare
        def long(*args,**kwargs):
            path,meta=original(*args,**kwargs); meta['retained_duration']=7200.0
            return path,meta
        self.c.sarvam_budget=150.0
        with patch.object(audio,'prepare',side_effect=long):
            self.assertEqual(self.run_pipeline()['completed'],2)
        self.assertEqual(len(self.fake.jobs),2)
        ledger=json.loads((Path(self.c.output_dir)/'.sarvam-budget.json').read_text())
        self.assertAlmostEqual(sum(ledger['reservations'].values()),141.12)

    def test_audio_file_is_uploaded_directly_when_already_compatible(self):
        self.c.audio_codec='wav'
        before=p.identity(self.source)
        self.assertEqual(self.run_pipeline()['completed'],2)
        self.assertEqual({path for path,_ in self.fake.uploads},{str(self.source)})
        self.assertTrue(self.manifest()['preprocessing']['source_audio_reused'])
        self.assertEqual(p.identity(self.source),before)


class AudioAndConfigurationTests(unittest.TestCase):
    def test_mov_extraction_and_audio_direct_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); mov=root/'தமிழ் video (1).mov'
            audio.command(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','color=c=black:s=32x32:r=1',
                '-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=2','-t','2',
                '-c:v','mpeg4','-c:a','pcm_s16le',str(mov)])
            before=sha256(mov)
            c=p.Config(audio_codec='flac',normalize_audio=False,preprocess_enabled=True)
            output,meta=audio.prepare(mov,root/'video-out',c,{'sha256':before})
            self.assertEqual(meta['input_kind'],'video'); self.assertFalse(meta['video_extraction_skipped'])
            self.assertEqual(meta['codec'],'flac'); self.assertEqual(meta['retained_duration'],2.0)
            self.assertEqual(sha256(mov),before)
            self.assertEqual(meta['audio_size_bytes'],output.stat().st_size)
            original_command=audio.command; calls=[]
            def tracked(args): calls.append(args); return original_command(args)
            with patch.object(audio,'command',side_effect=tracked):
                direct,details=audio.prepare(output,root/'audio-out',c,{'sha256':sha256(output)})
            self.assertEqual(direct,output)
            self.assertTrue(details['source_audio_reused']); self.assertTrue(details['video_extraction_skipped'])
            self.assertFalse(any(args[0]=='ffmpeg' and '-i' in args for args in calls))

    def test_nondefault_sample_rate_mapping_and_validation(self):
        mapping=audio.build_mapping([(0,8000),(16000,24000)],sample_rate=8000)
        self.assertEqual(audio.map_time(1,mapping),2)
        self.assertEqual(audio.map_segment(.5,1.5,mapping)[:2],(.5,2.5))
        p.validate(p.Config(audio_sample_rate=8000))
        with self.assertRaises(PipelineError): p.validate(p.Config(audio_sample_rate=44100))
        with self.assertRaises(PipelineError): p.validate(p.Config(audio_sample_rate=48000,vad_backend='silero',preprocess_enabled=True))

    def test_mapping_across_multiple_removed_intervals(self):
        mapping=audio.build_mapping([(0,3*16000),(10*16000,13*16000),(20*16000,23*16000)])
        start,end,spans=audio.map_segment(2.5,6.5,mapping)
        self.assertEqual((start,end),(2.5,20.5))
        self.assertEqual(spans,[{'start':2.5,'end':3.0},{'start':10.0,'end':13.0},{'start':20.0,'end':20.5}])

    def test_config_example_cli_precedence_and_unsafe_templates(self):
        c=p.load_config(p.parser().parse_args(['--config',str(p.ROOT/'config.example.toml'),
            '--audio-codec','wav','--audio-sample-rate','8000','--sarvam-budget','2','--only','translation']))
        self.assertEqual((c.audio_codec,c.audio_sample_rate,c.sarvam_budget),('wav',8000,2))
        self.assertFalse(c.transcript)
        self.assertEqual(p.output_name(c,Path('meeting.mov'),'translation'),'meeting.translate')
        self.assertEqual(p.output_path(c,Path('meeting.mov'),Path('meeting.mov')).name,'meeting')
        for directory in ('../{stem}','{stem.__class__}','/tmp','{stem!r}'):
            with self.assertRaises(PipelineError): p.validate(p.Config(output_directory=directory))
        for mode in ('transcribe','verbatim'):
            with self.assertRaises(PipelineError): p.validate(p.Config(recognition_mode=mode))

    def test_same_stem_collision_fails_before_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'meeting.mov').touch(); (root/'meeting.mp4').touch()
            with patch.object(p,'process',side_effect=AssertionError('must reject collision before work')):
                self.assertEqual(p.main(['--config',str(p.ROOT/'config.example.toml'),str(root),'--dry-run']),2)
