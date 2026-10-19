"""Native-media and budget regressions. Network is always replaced with a Batch double."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mozhiflow import audio_preprocess as a
from mozhiflow import pipeline as p
from mozhiflow import sarvam_provider as s
from mozhiflow.toolkit_core import atomic_json, sha256, fingerprint, PipelineError
from test_sarvam_dual import BatchDouble


class NativeAudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.c = p.Config(preprocess_enabled=True, normalize_audio=False, audio_codec='flac')

    def media(self, extension, codec, *, video=False):
        path = self.root / ('தமிழ் meeting (1).' + extension)
        args = ['ffmpeg', '-nostdin', '-v', 'error', '-y']
        if video:
            args += ['-f','lavfi','-i','color=c=black:s=32x32:r=1']
        args += ['-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=2', '-t','2', '-ac','2','-c:a',codec]
        if video: args += ['-c:v','mpeg4']
        a.command(args + [str(path)])
        return path

    def test_supported_formats_passthrough_identity_and_source_preservation(self):
        for extension, codec in [('m4a','aac'),('mp3','libmp3lame'),('wav','pcm_s24le'),
                                 ('flac','flac'),('aac','aac'),('ogg','libopus'),('opus','libopus')]:
            with self.subTest(extension=extension):
                source = self.media(extension, codec); before=p.identity(source)
                with patch.object(a, 'command', wraps=a.command) as calls:
                    path, meta = a.prepare(source,self.root/extension,self.c,before)
                self.assertEqual(path,source)
                self.assertFalse(any(call.args[0][0]=='ffmpeg' and '-i' in call.args[0] for call in calls.call_args_list))
                self.assertTrue(meta['audio_extraction_skipped']); self.assertTrue(meta['audio_conversion_skipped'])
                self.assertTrue(meta['source_audio_reused']); self.assertTrue(meta['source_audio_reused_reason'])
                self.assertEqual(a.map_segment(.1,1.5,meta['mapping'])[:2],(.1,1.5))
                self.assertEqual(p.identity(source),before)
                self.assertEqual(meta['sample_rate'],48000)

    def test_misleading_extension_uses_actual_streams(self):
        source=self.media('mp3','libmp3lame'); renamed=source.with_suffix('.m4a'); source.rename(renamed)
        path,meta=a.prepare(renamed,self.root/'out',self.c,p.identity(renamed))
        self.assertEqual(path,renamed); self.assertEqual(meta['upload_extension'],'mp3')
        self.assertEqual(s.upload_format(path),('mp3','audio/mpeg'))

    def test_unsupported_codec_converts_once(self):
        source=self.media('m4a','alac'); before=p.identity(source)
        path,meta=a.prepare(source,self.root/'out',self.c,before)
        self.assertNotEqual(path,source); self.assertEqual(path.suffix,'.flac')
        self.assertFalse(meta['audio_conversion_skipped']); self.assertIn('Unsupported',meta['audio_conversion_reason'])
        self.assertEqual(p.identity(source),before)

    def test_video_stream_even_in_audio_named_file_requires_extraction(self):
        source=self.media('mov','aac',video=True); renamed=source.with_suffix('.m4a'); source.rename(renamed)
        path,meta=a.prepare(renamed,self.root/'out',self.c,p.identity(renamed))
        self.assertFalse(meta['audio_extraction_skipped']); self.assertEqual(meta['input_kind'],'video')
        self.assertNotEqual(path,renamed)

    def test_vad_and_normalization_require_consolidated_conversion(self):
        source=self.media('m4a','aac')
        for operation in ('normalize','vad'):
            c=copy.copy(self.c); c.normalize_audio=operation=='normalize'; c.vad_backend='silero' if operation=='vad' else 'none'
            with patch.object(a,'detect_speech',return_value=([], 'mock')):
                path,meta=a.prepare(source,self.root/operation,c,p.identity(source))
            self.assertFalse(meta['source_audio_reused']); self.assertEqual(path.suffix,'.flac')
            self.assertTrue(meta['audio_extraction_skipped']); self.assertFalse(meta['audio_conversion_skipped'])

    def test_corrupt_source_rejected_locally(self):
        source=self.root/'broken.m4a'; source.write_bytes(b'not audio')
        with self.assertRaises(PipelineError): a.prepare(source,self.root/'out',self.c,p.identity(source))
        self.assertEqual(source.read_bytes(),b'not audio')

    def test_legacy_payload_preserved_for_paid_job_and_preview(self):
        source=self.media('m4a','aac'); src=p.identity(source)
        self.c.output_dir=str(self.root/'out'); self.c.output_directory='{stem}'
        directory=p.output_path(self.c,source,Path(source.name)); work=directory/'.preprocessing'/fingerprint(a.settings(self.c))
        work.mkdir(parents=True); converted=work/'processed.flac'
        a.command(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-ar','16000','-ac','1',str(converted)])
        meta={'source':src,'configuration':a.settings(self.c),'audio_sha256':sha256(converted),
              'codec':'flac','sample_rate':16000,'status':'success','input_kind':'audio','audio_path':str(converted.relative_to(directory)),
              'source_audio_reused':False,'retained_duration':2,'mapping':a.build_mapping([(0,32000)])}
        atomic_json(work/'preprocessing.json',meta)
        state=directory/'.providers/sarvam/token/remote.json'; state.parent.mkdir(parents=True)
        atomic_json(state,{'audio_sha256':meta['audio_sha256'],'phase':'started','job_id':'existing'})
        atomic_json(directory/'manifest.json',{'source':src,'preprocessing':meta})
        path,record=a.prepare(source,work,self.c,src)
        self.assertEqual(path,converted); self.assertIn('paid-job recovery',record['audio_conversion_reason'])
        preview=p.preview_audio(source,Path(source.name),self.c,src)
        self.assertEqual(preview['audio_sha256'],sha256(converted))


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.c=p.Config(output_dir=str(self.root),sarvam_budget=10)

    def job(self,token='new',recording='one',cost=3,mode='codemix',recoverable=False):
        return dict(token=token,recording=recording,cost=cost,mode=mode,recoverable=recoverable)

    def preflight(self,jobs,**kwargs): return s.budget_preflight(self.c,jobs,**kwargs)

    def test_combined_modes_all_or_nothing(self):
        jobs=[self.job(cost=6),self.job('translation',cost=6,mode='translate')]
        result=self.preflight(jobs)
        self.assertFalse(result['processing_permitted'])
        self.assertEqual(result['breakdown'][0]['additional_reservation_inr'],12)
        with self.assertRaises(s.BudgetPreflightError): self.preflight(jobs,reserve=True)
        self.assertFalse((self.root/'.sarvam-budget.json').exists())

    def test_recording_vs_invocation_vs_persistent_multiple_files(self):
        jobs=[self.job(cost=6),self.job('second',recording='two',cost=6)]
        for scope,permitted in [('persistent',False),('invocation',False),('recording',True)]:
            self.c.sarvam_budget_scope=scope
            self.assertEqual(self.preflight(jobs,invocation='run')['processing_permitted'],permitted)

    def test_committed_reserved_and_unknown_history_remain_counted(self):
        atomic_json(self.root/'.sarvam-budget.json',{'currency':'INR','reservations':{'done':4,'held':2,'failed':1}})
        for token,phase in [('done','downloaded'),('failed','started')]:
            path=self.root/'recording'/'.providers/sarvam'/token/'remote.json';path.parent.mkdir(parents=True)
            atomic_json(path,{'phase':phase,'remote_state':'Failed' if token=='failed' else 'Completed','job_id':token})
        row=self.preflight([self.job()])['breakdown'][0]
        self.assertEqual(row['previous_committed_cost_inr'],5)
        self.assertEqual(row['outstanding_reserved_cost_inr'],2)
        self.assertEqual(row['remaining_after_plan_inr'],0)
        self.preflight([self.job()],reserve=True)
        self.assertEqual(json.loads((self.root/'.sarvam-budget.json').read_text())['reservations']['done'],4)

    def test_persistent_history_excluded_only_by_explicit_invocation_scope(self):
        self.preflight([self.job('old',cost=8)],invocation='prior',reserve=True)
        self.assertFalse(self.preflight([self.job(cost=3)],invocation='next')['processing_permitted'])
        self.c.sarvam_budget_scope='invocation'
        self.assertTrue(self.preflight([self.job(cost=3)],invocation='next')['processing_permitted'])
        self.preflight([self.job(cost=3)],invocation='next',reserve=True)
        self.assertFalse(self.preflight([self.job('extra',cost=8)],invocation='next')['processing_permitted'])
        self.assertTrue(self.preflight([self.job('extra',cost=8)],invocation='later')['processing_permitted'])
        self.assertEqual(len(json.loads((self.root/'.sarvam-budget.json').read_text())['reservations']),2)

    def test_existing_job_recovery_with_lowered_cap(self):
        self.preflight([self.job('paid',cost=8)],reserve=True)
        self.c.sarvam_budget=0
        self.assertTrue(self.preflight([self.job('paid',cost=8,recoverable=True)])['processing_permitted'])
        self.assertFalse(self.preflight([self.job('fresh')])['processing_permitted'])
        s.reserve(self.c,'paid',8) # no new reservation or reset

    def test_config_scope_validation_and_override(self):
        config=self.root/'config.toml';config.write_text('[sarvam]\nbudget_inr=20\nbudget_scope="recording"\n')
        c=p.load_config(p.parser().parse_args(['--config',str(config),'--sarvam-budget-scope','invocation','--sarvam-budget','30']))
        self.assertEqual((c.sarvam_budget_scope,c.sarvam_budget),('invocation',30))
        config.write_text('[sarvam]\nbudget_inr=20\n')
        self.assertEqual(p.load_config(p.parser().parse_args(['--config',str(config)])).sarvam_budget_scope,'persistent')
        with self.assertRaises(PipelineError):p.validate(p.Config(sarvam_budget_scope='daily'))

    def test_main_blocks_entire_batch_before_any_paid_job(self):
        for n in (1,2):
            a.command(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','sine=duration=2',str(self.root/f'{n}.m4a')])
        config=self.root/'config.toml';config.write_text(f'''[paths]
input="{self.root}"
output_dir="{self.root}/outputs"
[recognition]
backend="sarvam"
model="saaras:v4"
[translation]
backend="sarvam"
[sarvam]
budget_inr=2.0
budget_scope="invocation"
[preprocessing]
normalize=false
[runtime]
offline=false
''')
        fake=BatchDouble();log=io.StringIO()
        with redirect_stdout(log),redirect_stderr(log),patch.object(s,'Transport',return_value=fake),patch.dict(os.environ,{'SARVAM_API_KEY':'unit-secret'}):
            self.assertEqual(p.main(['--config',str(config),'--allow-upload']),2)
        self.assertFalse(fake.jobs);self.assertIn('preflight blocked',log.getvalue())
        self.assertNotIn('RUN transcript',log.getvalue());self.assertNotIn('unit-secret',log.getvalue())
        self.assertFalse((self.root/'outputs/.sarvam-budget.json').exists())

    def test_native_m4a_submits_two_modes_then_recovers_without_new_calls(self):
        source=self.root/'meeting.m4a'
        a.command(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','sine=duration=2',str(source)])
        before=p.identity(source)
        c=p.Config(input=str(source),output_dir=str(self.root/'outputs'),recognition_backend='sarvam',model='saaras:v4',
                   translation_backend='sarvam',translation_model='saaras:v4',normalize_audio=False,offline=False,
                   upload_authorized=True,sarvam_budget=2,sarvam_budget_scope='invocation')
        fake=BatchDouble();log=io.StringIO()
        with redirect_stdout(log),redirect_stderr(log),patch.object(s,'Transport',return_value=fake),patch.dict(os.environ,{'SARVAM_API_KEY':'unit-secret'}):
            self.assertEqual(p.process(source,Path(source.name),c)['completed'],2)
            self.assertEqual(len(fake.jobs),2);self.assertEqual({Path(x[0]).resolve() for x in fake.uploads},{source.resolve()})
            create_calls=len(fake.calls);c.sarvam_budget=0;c.upload_authorized=False
            self.assertEqual(p.process(source,Path(source.name),c)['skipped'],2)
            self.assertEqual(len(fake.calls),create_calls)
        self.assertEqual(p.identity(source),before)
        for method,path,payload in fake.calls:
            if path=='/upload-files':self.assertEqual(payload['files'],['processed.m4a'])

    def test_main_authorized_batch_runs_exactly_two_jobs_and_retry_is_free(self):
        source=self.root/'meeting.m4a'
        a.command(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','sine=duration=2',str(source)])
        config=self.root/'config.toml';config.write_text(f'''[paths]
input="{source}"
output_dir="{self.root}/outputs"
[recognition]
backend="sarvam"
model="saaras:v4"
[translation]
backend="sarvam"
[sarvam]
budget_inr=2.0
budget_scope="invocation"
[preprocessing]
normalize=false
[runtime]
offline=false
''')
        fake=BatchDouble();log=io.StringIO()
        with redirect_stdout(log),redirect_stderr(log),patch.object(s,'Transport',return_value=fake),patch.dict(os.environ,{'SARVAM_API_KEY':'unit-secret'}):
            self.assertEqual(p.main(['--config',str(config),'--allow-upload']),0)
            self.assertEqual(len(fake.jobs),2)
            count=len(fake.calls)
            self.assertEqual(p.main(['--config',str(config)]),0)
            self.assertEqual(len(fake.calls),count)
        self.assertEqual({Path(x[0]).resolve() for x in fake.uploads},{source.resolve()})
        manifest=json.loads((self.root/'outputs/meeting.m4a/manifest.json').read_text())
        self.assertTrue(manifest['preprocessing']['audio_conversion_skipped'])
        self.assertEqual(manifest['stages']['transcript']['status'],'success')
        self.assertEqual(manifest['stages']['translation']['status'],'success')

    def test_already_reserved_unsubmitted_work_counts_against_invocation_cap(self):
        self.preflight([self.job(cost=8)],invocation='prior',reserve=True)
        self.c.sarvam_budget_scope='invocation';self.c.sarvam_budget=5
        self.assertFalse(self.preflight([self.job(cost=8)],invocation='next')['processing_permitted'])

    def test_failed_jobs_and_legacy_manifest_map_to_recording_scope(self):
        directory=self.root/'old';directory.mkdir()
        atomic_json(directory/'manifest.json',{'source':{'sha256':'one'}})
        state=directory/'.providers/sarvam/failed/remote.json';state.parent.mkdir(parents=True)
        atomic_json(state,{'phase':'started','remote_state':'Failed','job_id':'old-id'})
        atomic_json(self.root/'.sarvam-budget.json',{'currency':'INR','reservations':{'failed':9}})
        self.c.sarvam_budget_scope='recording'
        self.assertFalse(self.preflight([self.job(cost=2)])['processing_permitted'])
        self.assertTrue(self.preflight([self.job(recording='other',cost=2)])['processing_permitted'])
        self.assertEqual(json.loads(state.read_text())['job_id'],'old-id')
