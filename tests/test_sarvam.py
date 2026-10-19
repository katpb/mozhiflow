import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import Mock, MagicMock, patch

from mozhiflow.audio_preprocess import build_mapping, RATE
from mozhiflow import sarvam_provider as s
from mozhiflow.toolkit_core import PipelineError


def config(root):
    return SimpleNamespace(model='saaras:v4', language='ta', vocabulary=['API'],
        sarvam_diarization=False, sarvam_rate=0.49, sarvam_diarization_rate=0.59,
        sarvam_cost_margin=1.2, sarvam_budget=1.0, output_dir=str(root),
        upload_authorized=True, sarvam_job_id='', request_timeout=10.0,
        sarvam_poll_timeout=0.0, sarvam_poll_interval=0.01)


RAW = {'transcript': 'தமிழ் API 42', 'timestamps': {'chunks': ['தமிழ் API', '42'],
       'start_time_seconds': [0.0, 10.0], 'end_time_seconds': [1.0, 12.0]}, 'language_code': 'ta-IN'}
COMPLETE = {'job_state': 'Completed', 'failed_files_count': 0,
            'job_details': [{'state': 'Success', 'outputs': [{'file_name': '0.json'}]}]}


class SarvamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.c = config(self.root)
        self.audio = self.root / 'audio.wav'
        with wave.open(str(self.audio), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(b'\0\0' * 32000)
        self.prep = {'retained_duration': 20.0, 'audio_sha256': 'audiohash',
                     'mapping': build_mapping([(0, 10 * RATE), (20 * RATE, 30 * RATE)])}
        self.record = {}; self.save = Mock(); self.client = Mock()
        self.client.download.return_value = json.dumps(RAW).encode()
        self.env = patch.dict(os.environ, {'SARVAM_API_KEY': 'secret-key'})
        self.env.start(); self.addCleanup(self.env.stop)

    def run_provider(self):
        return s.SarvamRecognition().recognize(self.audio, self.prep, self.c, self.root, {'sha256': 'src'},
                                               {'backend': 'sarvam', 'model': 'saaras:v4'}, self.record, self.save,
                                               transport=self.client)

    def responses(self):
        self.client.api.side_effect = [{'job_id': 'job_1'}, {'storage_container_type': 'Azure_V1',
           'upload_urls': {'processed.wav': {'file_url': 'https://storage/upload'}}},
           {'job_state': 'Running'}, COMPLETE,
           {'download_urls': {'0.json': {'file_url': 'https://storage/download'}}}]

    def test_full_batch_and_raw_reuse(self):
        self.responses()
        result = self.run_provider()
        self.assertEqual(result['segments'][1]['start'], 20)
        self.assertEqual(result['segments'][1]['id'], 's000001')
        self.assertEqual(self.client.upload.call_count, 1)
        params = self.client.api.call_args_list[0].kwargs['payload']['job_parameters']
        self.assertEqual(params['mode'], 'codemix'); self.assertEqual(params['keyterms'], ['API'])
        self.assertTrue(params['with_timestamps'])
        raw = self.root / self.record['provider']['raw_response']
        self.assertEqual(raw.read_bytes(), json.dumps(RAW).encode())
        self.assertNotIn('secret-key', json.dumps(self.record))
        self.client.reset_mock(); self.c.upload_authorized = False
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.run_provider(), result)
        self.client.api.assert_not_called(); self.client.upload.assert_not_called()

    def test_missing_credential_consent_budget_before_network(self):
        for condition in ('key', 'consent', 'budget'):
            with self.subTest(condition=condition):
                self.c = config(self.root)
                self.c.upload_authorized = condition != 'consent'
                if condition == 'budget': self.c.sarvam_budget = 0.01
                with patch.dict(os.environ, {'SARVAM_API_KEY': '' if condition == 'key' else 'secret'}):
                    with self.assertRaises(PipelineError): self.run_provider()
                self.client.api.assert_not_called(); self.client.upload.assert_not_called()

    def test_poll_resume_does_not_create_start_or_upload_again(self):
        self.client.api.side_effect = [{'job_id': 'job_1'}, {'storage_container_type': 'Azure',
            'upload_urls': {'processed.wav': {'file_url': 'https://storage/upload'}}},
            {}, {'job_state': 'Running'}]
        with self.assertRaisesRegex(PipelineError, 'same job'):
            self.run_provider()
        self.client.reset_mock(); self.c.upload_authorized = False
        self.client.api.side_effect = [COMPLETE, {'download_urls': {'0.json': {'file_url': 'https://storage/download'}}}]
        self.run_provider()
        self.assertEqual(self.client.api.call_args_list[0].args, ('GET', '/job_1/status'))
        self.client.upload.assert_not_called()
        ledger = json.loads((self.root / '.sarvam-budget.json').read_text())
        self.assertEqual(len(ledger['reservations']), 1)

    def test_ambiguous_create_never_blindly_retried(self):
        self.client.api.side_effect = PipelineError('connection failed')
        with self.assertRaises(PipelineError): self.run_provider()
        self.client.reset_mock()
        with self.assertRaisesRegex(PipelineError, 'unknown outcome'): self.run_provider()
        self.client.api.assert_not_called()

    def test_ambiguous_start_resumes_existing_job(self):
        self.client.api.side_effect = [{'job_id': 'job_1'}, {'storage_container_type': 'Azure',
            'upload_urls': {'processed.wav': {'file_url': 'https://storage/upload'}}}, PipelineError('timeout')]
        with self.assertRaises(PipelineError): self.run_provider()
        self.client.reset_mock(); self.client.api.side_effect = [{'job_state': 'Accepted'}]
        with self.assertRaisesRegex(PipelineError, 'Start outcome is uncertain'): self.run_provider()
        self.client.upload.assert_not_called()
        self.assertEqual(self.client.api.call_count, 1)

    def test_failed_remote_job_never_replaced(self):
        self.client.api.side_effect = [{'job_id': 'job_1'}, {'storage_container_type': 'Azure',
            'upload_urls': {'processed.wav': {'file_url': 'https://storage/upload'}}}, {}, {'job_state': 'Failed'}]
        with self.assertRaisesRegex(PipelineError, 'no automatic replacement'): self.run_provider()
        self.client.api.side_effect = [{'job_state': 'Failed'}]; self.client.reset_mock()
        with self.assertRaises(PipelineError): self.run_provider()
        self.assertEqual(self.client.api.call_count, 1)

    def test_diarization_and_pricing_and_cumulative_budget(self):
        self.assertLess(s.estimate(25, self.c)['estimated_cost'], 1)
        normal = s.estimate(60, self.c)['estimated_cost']
        self.c.sarvam_diarization = True
        self.assertGreater(s.estimate(60, self.c)['estimated_cost'], normal)
        response = {'transcript': 'API', 'diarized_transcript': {'entries': [
            {'transcript': 'API', 'start_time_seconds': 1, 'end_time_seconds': 2, 'speaker_id': '0'}]}}
        self.assertEqual(s.normalize_response(response, self.prep['mapping'], diarization=True)['segments'][0]['speaker_id'], '0')
        s.reserve(self.c, 'jobA', 0.7); s.reserve(self.c, 'jobA', 0.7)
        with self.assertRaisesRegex(PipelineError, 'Cumulative'): s.reserve(self.c, 'jobB', 0.7)

    def test_invalid_response_retains_raw_no_reupload(self):
        self.responses(); self.client.download.return_value = b'{"transcript":"text"}'
        with self.assertRaisesRegex(PipelineError, 'timestamps'): self.run_provider()
        self.client.reset_mock()
        with self.assertRaises(PipelineError): self.run_provider()
        self.client.api.assert_not_called()
        self.assertTrue(list(self.root.rglob('provider-response.json')))

    def test_response_words_shape_and_invalid_alignment(self):
        raw = json.loads(json.dumps(RAW)); raw['timestamps']['words'] = raw['timestamps'].pop('chunks')
        self.assertEqual(len(s.normalize_response(raw, self.prep['mapping'])['segments']), 2)
        raw['timestamps']['end_time_seconds'].pop()
        with self.assertRaises(PipelineError): s.normalize_response(raw, self.prep['mapping'])

    def test_transport_does_not_leak_credentials_or_signed_urls(self):
        t = s.Transport('secret-key')
        t.opener = Mock()
        t.opener.open.side_effect = OSError('secret-key https://storage?token=secret')
        with self.assertRaises(PipelineError) as error:
            t.request('POST', s.API, payload={})
        self.assertNotIn('secret-key', str(error.exception))
        self.assertNotIn('token=', str(error.exception))
        with self.assertRaises(PipelineError): t.request('GET', 'https://evil.example')

    def test_response_import_is_source_bound_and_never_contacts_provider(self):
        response = self.root / 'existing.json'; response.write_text(json.dumps(RAW))
        self.c.sarvam_response_file = str(response); self.c.sarvam_response_source_sha256 = 'src'
        with patch.dict(os.environ, {}, clear=True):
            result = self.run_provider()
        self.assertEqual(result['text'], RAW['transcript'])
        self.assertEqual(self.record['provider']['origin'], 'user_supplied_response')
        self.assertEqual(self.record['provider']['cost']['estimated_cost'], 0)
        self.client.api.assert_not_called()
        self.client.upload.assert_not_called()
        self.run_provider()
        self.assertEqual(self.record['provider']['cost']['estimated_cost'], 0)

    def test_import_binding_mismatch_rejects_without_api(self):
        response = self.root / 'existing.json'; response.write_text(json.dumps(RAW))
        self.c.sarvam_response_file = str(response); self.c.sarvam_response_source_sha256 = 'wrong'
        with self.assertRaisesRegex(PipelineError, 'SHA-256'): self.run_provider()
        self.client.api.assert_not_called()

    def test_offline_and_duration_limits_fail_before_job_creation(self):
        self.c.offline = True
        with self.assertRaisesRegex(PipelineError, 'offline=true'): self.run_provider()
        self.c.offline = False; self.prep['retained_duration'] = 7201
        with self.assertRaisesRegex(PipelineError, '2 hours'): self.run_provider()
        self.client.api.assert_not_called()

    def test_output_format_changes_reuse_raw_result_not_billable_work(self):
        self.responses(); self.run_provider(); self.client.reset_mock()
        s.SarvamRecognition().recognize(self.audio,self.prep,self.c,self.root,{'sha256':'src'},
            {'backend':'sarvam','model':'saaras:v4','formats':['srt'],'filename':'different'},
            self.record,self.save,transport=self.client)
        self.client.api.assert_not_called()

    def test_local_model_preflight_failure_prevents_billable_creation(self):
        with self.assertRaisesRegex(PipelineError, 'local model missing'):
            s.SarvamRecognition().recognize(self.audio,self.prep,self.c,self.root,{'sha256':'src'},
                {'backend':'sarvam','model':'saaras:v4'},self.record,self.save,transport=self.client,
                before_upload=Mock(side_effect=PipelineError('local model missing')))
        self.client.api.assert_not_called()
        self.assertFalse((self.root / '.sarvam-budget.json').exists())

    def test_resume_id_on_fresh_state_never_creates_a_new_job(self):
        self.c.sarvam_job_id = 'job_existing'
        with self.assertRaisesRegex(PipelineError, 'refusing to create a replacement'):
            self.run_provider()
        self.client.api.assert_not_called()

    def test_signed_upload_omits_api_key_and_uses_blob_headers(self):
        transport = s.Transport('secret-key'); transport.opener = MagicMock()
        transport.opener.open.return_value.__enter__.return_value.read.return_value = b''
        transport.upload('https://storage.example/upload?token=private', self.audio, 'Azure_V1')
        req = transport.opener.open.call_args.args[0]
        headers = {k.lower(): v for k,v in req.header_items()}
        self.assertNotIn('api-subscription-key', headers)
        self.assertEqual(headers['x-ms-blob-type'], 'BlockBlob')
        self.assertEqual(headers['content-length'], str(self.audio.stat().st_size))
