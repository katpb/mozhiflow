"""Protect command compatibility and module identity across the package migration."""
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mozhiflow import __version__, pipeline


class PackagingTests(unittest.TestCase):
    def test_legacy_imports_share_module_identity_and_patches(self):
        for name in ('pipeline', 'audio_preprocess', 'providers', 'sarvam_provider',
                     'ollama_provider', 'toolkit_core'):
            legacy = importlib.import_module(name)
            packaged = importlib.import_module(f'mozhiflow.{name}')
            self.assertIs(legacy, packaged)
        with patch('pipeline.inference') as inference:
            self.assertIs(pipeline.inference, inference)

    def test_entry_points_have_the_same_help_and_version(self):
        root = Path(__file__).resolve().parents[1]
        commands = ([sys.executable, '-m', 'mozhiflow'],
                    [sys.executable, str(root / 'pipeline.py')],
                    [str(root / 'transcribe.sh')])
        for option in ('--help', '--version'):
            outputs = []
            for command in commands:
                result = subprocess.run([*command, option], cwd=root, text=True,
                                        capture_output=True, check=True)
                outputs.append(result.stdout)
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[0], outputs[2])
            self.assertIn('mozhiflow' if option == '--help' else __version__, outputs[0])

    def test_worker_bootstraps_outside_checkout_without_optional_dependencies(self):
        worker = Path(pipeline.__file__).with_name('_worker.py')
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, str(worker)], input='{}', cwd=cwd,
                                    text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("KeyError: 'source'", result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)
