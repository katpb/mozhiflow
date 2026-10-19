#!/usr/bin/env python3
"""Run real MLX validation plus safe retry checks. Never modifies source media."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mozhiflow import pipeline as p


def validate(config, report):
    c = p.load_config(p.parser().parse_args(['--config', str(config)]))
    if not c.transcript or not c.translation or set(c.formats) != {'txt', 'srt', 'json'} or c.overwrite:
        raise p.PipelineError('Validation requires both stages, TXT/SRT/JSON, and overwrite=false')
    files = p.discover(c)
    before = {str(source): p.identity(source) for source, _ in files}
    command = [sys.executable, '-m', 'mozhiflow', '--config', str(config.resolve())]
    result = subprocess.run(command, text=True, capture_output=True)
    evidence = {'configuration': str(config.resolve()), 'command': command, 'exit_code': result.returncode,
                'source_unchanged': all(p.identity(Path(path)) == src for path, src in before.items()),
                'stdout': result.stdout, 'stderr': result.stderr, 'validated': False}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    if result.returncode or not evidence['source_unchanged']:
        raise p.PipelineError(f'Real validation did not pass; see {report}')
    artifacts, attempts, stats = {}, {}, []
    for source, relative in files:
        directory = p.output_path(c, source, relative)
        manifest = json.loads((directory / 'manifest.json').read_text())
        for stage in p.STAGES:
            record = manifest['stages'][stage]
            if record['status'] != 'success' or not p.intact(record, directory):
                raise p.PipelineError(f'Incomplete/corrupt stage: {source} {stage}')
            attempts[str(directory), stage] = record['attempts']
            prefix = p.output_name(c, source, stage)
            for fmt in c.formats:
                path = directory / f'{prefix}.{fmt}'
                if not path.stat().st_size:
                    raise p.PipelineError(f'Empty output: {path}')
                artifacts[str(path)] = p.digest(path)
            decoded = json.loads((directory / f'{prefix}.json').read_text())
            p.validate_result(decoded)
            if (directory / f'{prefix}.txt').read_text() != decoded['text']:
                raise p.PipelineError('TXT differs from raw JSON text')
            if (directory / f'{prefix}.srt').read_text() != p.render(decoded, 'srt'):
                raise p.PipelineError('SRT differs from raw JSON timestamps')
            stats.append({'file': source.name, 'stage': stage, 'segments': len(decoded['segments']),
                          'characters': len(decoded['text']), 'last_end': decoded['segments'][-1]['end']})
    # Neither an unchanged rerun nor translation-only retry should repeat successful inference.
    for extra in ([], ['--only', 'translation']):
        retry = subprocess.run(command + extra, text=True, capture_output=True)
        if retry.returncode:
            raise p.PipelineError(f'Retry failed: {retry.stderr}')
        for (directory, stage), count in attempts.items():
            current = json.loads((Path(directory) / 'manifest.json').read_text())
            if current['stages'][stage]['attempts'] != count:
                raise p.PipelineError(f'Retry repeated a successful {stage} stage')
        if any(p.digest(Path(path)) != sha for path, sha in artifacts.items()):
            raise p.PipelineError('Retry changed an output')
    evidence.update(validated=True, outputs=stats, retry_preserved_attempts=True, retry_preserved_artifacts=True)
    report.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(f'Real validation passed: {report}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--report', type=Path, default=p.ROOT / 'validation/private/evidence.json')
    args = parser.parse_args()
    try:
        validate(args.config, args.report)
    except (p.PipelineError, OSError, ValueError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        raise SystemExit(1)
