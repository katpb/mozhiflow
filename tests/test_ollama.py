import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from mozhiflow import ollama_provider as o
from mozhiflow.toolkit_core import PipelineError


def config():
    return SimpleNamespace(ollama_endpoint='http://127.0.0.1:11434', request_timeout=20,
        translation_model='gemma3:12b', ollama_options={'temperature': 0.0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 2048},
        translation_group_segments=1, translation_context_segments=1, translation_max_chars=4000,
        translation_retries=1, translation_prompt='', vocabulary=['API'], overwrite=False)


TRANSCRIPT = {'text': 'API 42 வேண்டாம். Redis இருந்தால் மட்டும்.', 'segments': [
    {'id': 's000000', 'start': 5, 'end': 7, 'text': 'API 42 வேண்டாம்.'},
    {'id': 's000001', 'start': 20, 'end': 22, 'text': 'Redis இருந்தால் மட்டும்.'}]}


class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.c = config(); self.record = {}; self.save = Mock()
        self.client = Mock()
        def request(path, payload=None):
            if path == '/api/tags': return {'models': [{'name': self.c.translation_model, 'digest': 'digest1'}]}
            data = json.loads(payload['prompt'].split('\nPrevious')[0])
            text = {'s000000': 'Do not use API 42.', 's000001': 'Only if Redis is available.'}
            return {'done': True, 'done_reason': 'stop', 'response': json.dumps({'segments': [
                {'id': s['id'], 'text': text[s['id']]} for s in data['targets']]})}
        self.client.request.side_effect = request

    def run_provider(self):
        return o.OllamaTranslation().translate(TRANSCRIPT, self.c, self.root, self.record, self.save, client=self.client)

    def test_translation_keeps_ids_timestamps_and_context(self):
        before = copy.deepcopy(TRANSCRIPT)
        result = self.run_provider()
        for a, b in zip(result['segments'], before['segments']):
            self.assertEqual((a['id'], a['start'], a['end']), (b['id'], b['start'], b['end']))
        self.assertEqual(TRANSCRIPT, before)
        request = self.client.request.call_args_list[1].args[1]
        self.assertIn('neighbor_context', request['prompt']); self.assertFalse(request['stream'])
        self.assertEqual(self.record['provider']['completed_groups'], 2)
        self.assertIn('negation', request['system'])
        self.client.reset_mock(); self.run_provider()
        self.assertEqual(self.client.request.call_count, 1)  # tags only; groups reused

    def test_model_missing_and_nonlocal_endpoints(self):
        for endpoint in ['https://example.com', 'http://127.0.0.1@evil.example', 'http://localhost/api', 'file:///tmp/model']:
            with self.assertRaises(PipelineError): o.validate_endpoint(endpoint)
        self.client.request.side_effect = None; self.client.request.return_value = {'models': []}
        with self.assertRaisesRegex(PipelineError, 'not installed'): self.run_provider()

    def test_loopback_server_cannot_substitute_a_cloud_model(self):
        self.client.request.side_effect = None
        self.client.request.return_value = {'models': [{'name': self.c.translation_model, 'digest': 'digest',
                                                        'remote_model': 'gemma3:12b', 'remote_host': 'https://ollama.com'}]}
        with self.assertRaisesRegex(PipelineError, 'remote/cloud'): self.run_provider()
        self.assertEqual(self.client.request.call_count, 1)

    def test_bad_ids_numbers_identifiers_or_extra_fields(self):
        source = [{'id': 'id', 'text': 'API 42 user_id v2'}]
        for response in ({'segments': [{'id': 'bad', 'text': 'API 42 user_id v2'}]},
                         {'segments': [{'id': 'id', 'text': 'API 43 user_id v2'}]},
                         {'segments': [{'id': 'id', 'text': '42 user_id v2'}]},
                         {'segments': [{'id': 'id', 'text': 'API 42 user_id v2', 'start': 0}]},
                         {'segments': []}):
            with self.assertRaises(PipelineError): o.validate_translation(response, source, [])

    def test_bounded_groups_reject_oversized_segment(self):
        self.c.translation_max_chars = 4
        with self.assertRaises(PipelineError): self.run_provider()

    def test_failed_group_resumes_without_repeating_completed_group(self):
        original = self.client.request.side_effect
        def fail_second(path, payload=None):
            if path == '/api/generate' and 's000001' in json.loads(payload['prompt'].split('\nPrevious')[0])['targets'][0]['id']:
                return {'done': True, 'response': '{bad'}
            return original(path, payload)
        self.client.request.side_effect = fail_second
        with self.assertRaisesRegex(PipelineError, 'successful groups retained'): self.run_provider()
        self.assertEqual(self.record['provider']['completed_groups'], 1)
        self.client.reset_mock(); self.client.request.side_effect = original
        self.run_provider()
        self.assertEqual(self.client.request.call_count, 2)  # tags + failed group only

    def test_prompt_and_model_digest_changes_invalidate_groups(self):
        self.run_provider(); self.client.reset_mock()
        self.c.translation_prompt = 'Keep colloquial tone.'
        self.run_provider(); self.assertEqual(self.client.request.call_count, 3)

    def test_truncated_generation_rejected(self):
        original = self.client.request.side_effect
        self.client.request.side_effect = lambda path, payload=None: (
            {'done': True, 'done_reason': 'length', 'response': '{}'} if path == '/api/generate' else original(path, payload))
        with self.assertRaisesRegex(PipelineError, 'truncated'): self.run_provider()
