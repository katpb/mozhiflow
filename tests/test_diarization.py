"""Diarization request, timestamp, rendering, replay and no-paid-replacement regressions."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from mozhiflow import audio_preprocess as a
from mozhiflow import pipeline as p
from mozhiflow import sarvam_provider as s
from mozhiflow.toolkit_core import PipelineError, sha256
import test_sarvam_dual as dual


class DiarizationTests(unittest.TestCase):
    setUp = dual.DualSarvamTests.setUp
    directory = dual.DualSarvamTests.directory
    manifest = dual.DualSarvamTests.manifest
    run_pipeline = dual.DualSarvamTests.run_pipeline

    def enable(self):
        self.c.sarvam_diarization=True
        self.c.sarvam_speaker_names={'codemix':{'0':'Kathir'},'translate':{'1':'Kathir'}}
        def download(url):
            mode=self.fake.jobs[url.split('/')[-1]]
            speaker='0' if mode=='codemix' else '1'
            text='தமிழ் API 42' if mode=='codemix' else 'Use API 42.'
            return json.dumps({'transcript':text+' '+text,'language_code':'ta-IN',
                'diarized_transcript':{'entries':[
                    {'speaker_id':speaker,'start_time_seconds':.1,'end_time_seconds':.8,'transcript':text},
                    {'speaker_id':None,'start_time_seconds':1,'end_time_seconds':1.8,'transcript':text}]}}).encode()
        self.fake.download=download

    def test_both_modes_render_speakers_and_timestamps_and_preserve_raw(self):
        self.enable();self.c.sarvam_num_speakers=2
        self.assertEqual(self.run_pipeline()['completed'],2)
        params=[payload['job_parameters'] for method,path,payload in self.fake.calls if method=='POST' and path=='']
        self.assertEqual(len(params),2)
        self.assertTrue(all(x['with_diarization'] and x['with_timestamps'] and x['num_speakers']==2 for x in params))
        m=self.manifest()
        for stage,mode,original_id in [('transcript','codemix','0'),('translation','translate','1')]:
            result=json.loads((self.directory/f'{self.source.stem}.{mode}.json').read_text())
            self.assertEqual(result['segments'][0]['speaker_id'],original_id)
            self.assertEqual(result['segments'][0]['speaker_name'],'Kathir')
            self.assertEqual(result['segments'][1]['speaker_status'],'unknown')
            self.assertIsNone(result['segments'][1]['speaker_id'])
            self.assertEqual(result['speaker_namespace']['mode'],mode)
            self.assertEqual(result['speaker_namespace']['job_id'],m['stages'][stage]['provider']['remote_job_id'])
            txt=(self.directory/f'{self.source.stem}.{mode}.txt').read_text()
            srt=(self.directory/f'{self.source.stem}.{mode}.srt').read_text()
            self.assertIn('[00:00:00,100 --> 00:00:00,800] Kathir:',txt)
            self.assertIn(f'{mode} unknown speaker:',srt)
            raw=self.directory/m['stages'][stage]['provider']['raw_response']
            self.assertEqual(sha256(raw),m['stages'][stage]['provider']['raw_response_sha256'])
        calls=len(self.fake.calls);self.c.upload_authorized=False
        self.assertEqual(self.run_pipeline()['skipped'],2);self.assertEqual(len(self.fake.calls),calls)

    def test_auto_detection_omits_num_speakers(self):
        self.enable();self.assertEqual(self.run_pipeline()['completed'],2)
        for method,path,payload in self.fake.calls:
            if method=='POST' and path=='':self.assertNotIn('num_speakers',payload['job_parameters'])

    def test_name_changes_replay_without_network_and_keep_ids(self):
        self.enable();self.run_pipeline()
        raw_before={path:sha256(path) for path in self.directory.glob('.providers/sarvam/*/provider-response.json')}
        calls=len(self.fake.calls)
        self.c.overwrite=True;self.c.upload_authorized=False
        self.c.sarvam_speaker_names={'codemix':{'0':'Reviewed name'}}
        self.assertEqual(self.run_pipeline()['completed'],2)
        self.assertEqual(len(self.fake.calls),calls)
        result=json.loads((self.directory/f'{self.source.stem}.codemix.json').read_text())
        self.assertEqual(result['segments'][0]['speaker_id'],'0')
        self.assertEqual(result['segments'][0]['speaker_name'],'Reviewed name')
        translated=json.loads((self.directory/f'{self.source.stem}.translate.json').read_text())
        self.assertEqual(translated['segments'][0]['speaker_label'],'translate speaker 1')
        self.assertEqual({path:sha256(path) for path in raw_before},raw_before)

    def test_no_automatic_replacement_of_successful_or_failed_submitted_jobs(self):
        for failed in (False,True):
            # Separate subdirectories isolate paid history in each scenario.
            self.c.output_dir=str(self.root/str(failed));self.c.sarvam_diarization=False
            self.c.overwrite=False
            self.fake=dual.BatchDouble()
            if failed:self.fake.failed={'codemix','translate'}
            self.run_pipeline()
            before={path:sha256(path) for path in self.directory.rglob('*') if path.is_file() and path.name!='.pipeline.lock'}
            calls=len(self.fake.calls);self.c.sarvam_diarization=True;self.c.overwrite=True
            with self.assertRaisesRegex(PipelineError,'No replacement job'):
                self.run_pipeline()
            self.assertEqual(len(self.fake.calls),calls)
            for path,digest in before.items():
                if path.name not in {'manifest.json','preprocessing.json'}:self.assertEqual(sha256(path),digest)
            # No new ledger reservation despite --overwrite and retained upload permission.
            self.assertEqual(len(json.loads((Path(self.c.output_dir)/'.sarvam-budget.json').read_text())['reservations']),2)

    def test_count_change_preserves_submitted_job(self):
        self.enable();self.run_pipeline();calls=len(self.fake.calls)
        self.c.sarvam_num_speakers=3
        with self.assertRaisesRegex(PipelineError,'No replacement job'):self.run_pipeline()
        self.assertEqual(len(self.fake.calls),calls)

    def test_missing_diarization_retains_raw_without_replacement_on_retry(self):
        self.c.sarvam_diarization=True
        self.assertEqual(self.run_pipeline()['failed'],2)
        calls=len(self.fake.calls)
        self.assertEqual(self.run_pipeline()['failed'],2)
        self.assertEqual(len(self.fake.calls),calls)
        self.assertEqual(len(list(self.directory.glob('.providers/sarvam/*/provider-response.json'))),2)

    def test_combined_diarized_budget_blocks_before_first_job(self):
        self.enable();self.c.sarvam_budget=1.5
        with self.assertRaises(s.BudgetPreflightError):self.run_pipeline()
        self.assertFalse(self.fake.jobs)
        row=self.manifest()['budget_preflight']['breakdown'][0]
        self.assertEqual(row['additional_reservation_inr'],1.8)

    def test_remote_history_protects_failed_stage_even_without_manifest_provider(self):
        self.c.translation=False;self.fake.failed.add('codemix')
        self.assertEqual(self.run_pipeline()['failed'],1)
        manifest=self.manifest();manifest['stages']['transcript'].pop('provider')
        (self.directory/'manifest.json').write_text(json.dumps(manifest))
        calls=len(self.fake.calls);self.c.sarvam_diarization=True
        with self.assertRaisesRegex(PipelineError,'No replacement job'):self.run_pipeline()
        self.assertEqual(len(self.fake.calls),calls)

    def test_ambiguous_create_cannot_be_replaced_by_changing_diarization(self):
        self.c.translation=False;self.fake.interrupt_create='codemix'
        with self.assertRaises(KeyboardInterrupt):self.run_pipeline()
        calls=len(self.fake.calls);self.c.sarvam_diarization=True
        with self.assertRaisesRegex(PipelineError,'No replacement job'):self.run_pipeline()
        self.assertEqual(len(self.fake.calls),calls)


class DiarizationParsingTests(unittest.TestCase):
    def test_vad_mapping_overlaps_unknown_and_uncertainty(self):
        mapping=a.build_mapping([(0,16000),(160000,176000)])
        raw={'transcript':'API 42 No API','diarized_transcript':{'entries':[
            {'speaker_id':'0','transcript':'API 42','start_time_seconds':.5,'end_time_seconds':1.5},
            {'speaker_id':'0','speaker_uncertain':True,'speaker_confidence':0,
             'transcript':'No API','start_time_seconds':1,'end_time_seconds':1.8}]}}
        result=s.normalize_response(raw,mapping,diarization=True,speaker_names={'0':'Named'})
        self.assertEqual((result['segments'][0]['start'],result['segments'][0]['end']),(.5,10.5))
        self.assertEqual(len(result['segments'][0]['original_spans']),2)
        self.assertEqual(result['segments'][1]['speaker_status'],'uncertain')
        self.assertIsNone(result['segments'][1]['speaker_name'])
        self.assertEqual(result['segments'][1]['speaker_id'],'0')
        self.assertIn('uncertain speaker',p.render(result,'txt'))

    def test_missing_timestamps_not_fabricated_and_malformed_entries_rejected(self):
        raw={'transcript':'text','diarized_transcript':{'entries':[{'transcript':'text'}]}}
        result=s.normalize_response(raw,a.build_mapping([(0,16000)]),diarization=True,timestamp_policy='flag')
        self.assertIsNone(result['segments'][0]['start'])
        self.assertIn('timestamp unavailable',p.render(result,'txt'));self.assertEqual(p.render(result,'srt'),'')
        for entries in ([None],[],{'speaker_id':'0'}):
            raw['diarized_transcript']['entries']=entries
            with self.assertRaises(PipelineError):s.normalize_response(raw,a.build_mapping([(0,16000)]),diarization=True)

    def test_configuration_validation_alias_and_optional_count(self):
        for count in (1,20):p.validate(p.Config(sarvam_diarization=True,sarvam_num_speakers=count))
        for count in (0,21,True,'2',2.0):
            with self.assertRaises(PipelineError):p.validate(p.Config(sarvam_diarization=True,sarvam_num_speakers=count))
        with self.assertRaises(PipelineError):p.validate(p.Config(sarvam_num_speakers=2))
        with self.assertRaises(PipelineError):p.validate(p.Config(sarvam_diarization=True,sarvam_timestamps=False))
        for names in ({'0':'name'},{'codemix':{'0':'bad\nname'}},{'translate':{'0':42}}):
            with self.assertRaises(PipelineError):p.validate(p.Config(sarvam_speaker_names=names))
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'c.toml';config.write_text('[sarvam]\nwith_diarization=true\nnum_speakers=2\n[sarvam.speaker_names.codemix]\n"0"="Name"\n')
            c=p.load_config(p.parser().parse_args(['--config',str(config)]))
            self.assertTrue(c.sarvam_diarization);self.assertEqual(c.sarvam_num_speakers,2)
            with config.open('w') as f:f.write('[sarvam]\nwith_diarization=true\ndiarization=false\n')
            with self.assertRaisesRegex(PipelineError,'Conflicting'):p.load_config(p.parser().parse_args(['--config',str(config)]))

    def test_price_and_render_only_names_do_not_change_paid_token(self):
        c=p.Config(sarvam_diarization=True)
        estimate=s.estimate(3600,c)
        self.assertEqual(estimate['rate_per_minute'],.75)
        self.assertEqual(estimate['estimated_cost']*2,108)
        settings=p.stage_settings(c,'transcript');settings['backend']='sarvam'
        before=s.job_token({'sha256':'source'},settings,{'audio_sha256':'audio'})
        settings['speaker_names']={'0':'Name'}
        self.assertEqual(before,s.job_token({'sha256':'source'},settings,{'audio_sha256':'audio'}))
        settings['num_speakers']=2
        self.assertNotEqual(before,s.job_token({'sha256':'source'},settings,{'audio_sha256':'audio'}))

    def test_same_provider_id_in_different_modes_has_no_implicit_name_link(self):
        raw={'transcript':'API','diarized_transcript':{'entries':[
            {'speaker_id':'0','transcript':'API','start_time_seconds':0,'end_time_seconds':1}]}}
        mapping=a.build_mapping([(0,16000)])
        source=s.normalize_response(raw,mapping,diarization=True,mode='codemix',speaker_names={'0':'Reviewed'})
        target=s.normalize_response(raw,mapping,diarization=True,mode='translate')
        self.assertEqual(source['segments'][0]['speaker_name'],'Reviewed')
        self.assertIsNone(target['segments'][0]['speaker_name'])
        self.assertEqual(target['segments'][0]['speaker_label'],'translate speaker 0')
