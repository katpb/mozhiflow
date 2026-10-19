#!/usr/bin/env python3
"""Explicit short provider validation through the existing CLI; optional saved-reference comparison."""
import argparse
import difflib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mozhiflow import pipeline as p
from mozhiflow.toolkit_core import atomic_json, sha256


def run(config, reference, report, allow_upload=False):
    c = p.load_config(p.parser().parse_args(['--config', str(config)]))
    files = p.discover(c)
    if len(files) != 1 or c.recognition_backend != 'sarvam' or c.translation_backend not in {'ollama', 'sarvam'} or c.overwrite:
        raise p.PipelineError('Use one short Saaras sample with sarvam/ollama translation, overwrite=false, and both stages')
    if not c.transcript or not c.translation or set(c.formats) != {'txt', 'srt', 'json'}:
        raise p.PipelineError('Sample validation requires both outputs in TXT/SRT/JSON')
    source, relative = files[0]
    from mozhiflow.audio_preprocess import probe
    duration, _ = probe(source)
    if duration > 30:
        raise p.PipelineError('This validator accepts a sample of at most 30 seconds, never a whole meeting')
    before = p.identity(source)
    command = [sys.executable, '-m', 'mozhiflow', '--config', str(config.resolve())]
    if allow_upload:
        command.append('--allow-upload')
    result = subprocess.run(command, capture_output=True, text=True)
    evidence = {'configuration': str(config.resolve()), 'source': before, 'duration': duration,
                'source_unchanged': p.identity(source) == before, 'exit_code': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr, 'outputs_validated': False,
                'translation_quality_verified': False, 'recognition_mode': 'import' if c.sarvam_response_file else 'live_batch',
                'translation_provider': c.translation_backend,
                'translation_mode': 'import' if c.sarvam_translation_response_file else 'provider'}
    atomic_json(report, evidence)
    if result.returncode:
        raise p.PipelineError(f'Configured provider run did not complete; diagnostics saved to {report}')
    directory = p.output_path(c, source, relative)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    outputs = {}
    attempts = {s: manifest['stages'][s]['attempts'] for s in p.STAGES}
    hashes = {}
    for stage in p.STAGES:
        record = manifest['stages'][stage]
        if record['status'] != 'success' or not p.intact(record, directory):
            raise p.PipelineError(f'{stage} is incomplete')
        prefix = p.output_name(c, source, stage)
        for fmt in c.formats:
            path = directory / f'{prefix}.{fmt}'
            if not path.stat().st_size:
                raise p.PipelineError(f'Empty {stage} {fmt}')
            hashes[str(path)] = sha256(path)
        outputs[stage] = json.loads((directory / f'{prefix}.json').read_text())
        p.validate_result(outputs[stage])
    source_segments, translated_segments = outputs['transcript']['segments'], outputs['translation']['segments']
    if c.translation_backend == 'ollama' and len(source_segments) != len(translated_segments):
        raise p.PipelineError('Translation segment alignment differs from transcript')
    for original, translated in (zip(source_segments, translated_segments) if c.translation_backend == 'ollama' else []):
        if any(original[key] != translated[key] for key in ('id', 'start', 'end')):
            raise p.PipelineError('Translation changed IDs or original timestamps')
    retry = subprocess.run(command + ['--only', 'translation'], capture_output=True, text=True)
    after = json.loads(manifest_path.read_text())
    if retry.returncode or any(after['stages'][s]['attempts'] != attempts[s] for s in p.STAGES):
        raise p.PipelineError('A successful stage was unnecessarily reprocessed')
    if any(sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise p.PipelineError('Retry altered output artifacts')
    english = outputs['translation']['text']
    diff = None
    if reference.is_file():
        baseline = json.loads(reference.read_text())['transcript']
        diff = '\n'.join(difflib.unified_diff(baseline.splitlines(), english.splitlines(),
                            fromfile='previous_saaras_translation', tofile='current_translation', lineterm=''))
    evidence.update(outputs_validated=True, translation_retry_skipped=True, output_hashes=hashes,
                    manifest=str(manifest_path), providers={s: after['stages'][s].get('provider') for s in p.STAGES},
                    reference_sha256=sha256(reference) if reference.is_file() else None, textual_comparison=diff,
                    quality_review='Textual differences require human listening for meaning, negation, numbers, conditions and uncertainty; no semantic equivalence claim')
    atomic_json(report, evidence)
    print(f'Output integrity and retry validation passed; quality review remains required: {report}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=p.ROOT / 'config.toml')
    ap.add_argument('--reference', type=Path, default=p.ROOT / 'saaras-test/translate.json')
    ap.add_argument('--report', type=Path, default=p.ROOT / 'validation/private/hybrid-evidence.json')
    ap.add_argument('--allow-upload', action='store_true', help='Explicitly authorize ONLY this short sample upload')
    args = ap.parse_args()
    try:
        run(args.config, args.reference, args.report, args.allow_upload)
    except (p.PipelineError, OSError, ValueError, KeyError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        raise SystemExit(1)
