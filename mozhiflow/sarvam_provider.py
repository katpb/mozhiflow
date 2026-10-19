"""Official Saaras V4 Batch protocol with durable, conservative remote resumption."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .audio_preprocess import map_segment
from .toolkit_core import PipelineError, atomic_json, atomic_bytes, fingerprint, sha256

API = 'https://api.sarvam.ai/speech-to-text/job/v1'


class SarvamHTTPError(PipelineError):
    def __init__(self, status):
        self.status = status
        super().__init__(f'Sarvam HTTP {status}; authentication/quota/rate-limit or provider error; remote state retained')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PipelineError('Provider redirect refused; no credentials forwarded')


class Transport:
    def __init__(self, key, timeout=60, read_retries=2):
        self.key, self.timeout, self.read_retries = key, timeout, read_retries
        self.opener = urllib.request.build_opener(NoRedirect)

    def request(self, method, url, *, payload=None, data=None, signed=False, headers=None):
        if not url.startswith('https://'):
            raise PipelineError('Sarvam and signed storage URLs must use HTTPS')
        h = dict(headers or {})
        if not signed:
            if not url.startswith(API):
                raise PipelineError('Refusing to send credentials to a non-Sarvam endpoint')
            h['api-subscription-key'] = self.key
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode()
            h['Content-Type'] = 'application/json'
        try:
            req = urllib.request.Request(url, data=data, headers=h, method=method)
            with self.opener.open(req, timeout=self.timeout) as response:
                body = response.read()
            return body if signed else json.loads(body)
        except urllib.error.HTTPError as e:
            # Never print request headers, response bodies, exception URLs or signed tokens.
            raise SarvamHTTPError(e.code) from None
        except (OSError, ValueError, urllib.error.URLError):
            raise PipelineError('Sarvam connection/response failure; remote state retained; credentials and URLs suppressed') from None

    def api(self, method, path='', payload=None):
        # Only reads are automatically retried. Ambiguous billable mutations are never retried here.
        for attempt in range(self.read_retries + 1):
            try:
                return self.request(method, API + path, payload=payload)
            except PipelineError as e:
                if isinstance(e, SarvamHTTPError) and e.status not in {429, 500, 502, 503, 504}:
                    raise
                if method != 'GET' or attempt == self.read_retries:
                    raise
                time.sleep(2 ** attempt)

    def upload(self, url, path, storage):
        headers = {'Content-Type': upload_format(path)[1],
                   'Content-Length': str(path.stat().st_size)}
        if storage in {'Azure', 'Azure_V1'}:
            headers['x-ms-blob-type'] = 'BlockBlob'
        with path.open('rb') as stream:
            self.request('PUT', url, data=stream, signed=True, headers=headers)

    def download(self, url):
        return self.request('GET', url, signed=True)


def upload_format(path):
    from .audio_preprocess import media_info, compatible_format
    _, stream = media_info(path)
    supported = compatible_format(stream)
    if supported:
        return supported
    raise PipelineError('No supported upload codec/container')


def estimate(seconds, c):
    rate = c.sarvam_diarization_rate if c.sarvam_diarization else c.sarvam_rate
    # Whole-minute rounding plus a configurable margin is a conservative local estimate, not a billing promise.
    return {'currency': 'INR', 'duration_seconds': seconds, 'rate_per_minute': rate,
            'diarization': c.sarvam_diarization, 'rounded_minutes': math.ceil(seconds / 60),
            'estimated_cost': round(math.ceil(seconds / 60) * rate * c.sarvam_cost_margin, 6),
            'margin': c.sarvam_cost_margin, 'pricing_verified': '2026-10-19',
            'pricing_source': ('https://docs.sarvam.ai/api/getting-started/pricing' if c.sarvam_diarization
                               else 'https://www.sarvam.ai/api-pricing'),
            'note': 'Local estimate only; diarization default uses conservative INR45/hour because official pages disagree; excludes tax and billing changes'}


class BudgetPreflightError(PipelineError):
    pass


def budget_preflight(c, jobs, *, invocation='', reserve=False):
    """Read-only preview or locked all-or-nothing reservation of the entire plan.

    Legacy reservations are never discarded. Unknown/failed/ambiguous work stays
    reserved conservatively; started jobs are committed estimates, not account usage.
    """
    root = Path(c.output_dir)
    ledger = root / '.sarvam-budget.json'
    scope = getattr(c, 'sarvam_budget_scope', 'persistent')

    def calculate(data):
        reservations = data.get('reservations', {})
        details = data.get('entries', {})
        if (not isinstance(reservations, dict) or any(type(cost) not in (float, int) or not math.isfinite(cost) or cost < 0
                                                    for cost in reservations.values())):
            raise PipelineError('Invalid local budget ledger; history retained, refusing new submissions')
        states = {}
        recordings = {}
        for manifest_path in root.glob('**/manifest.json'):
            manifest = json.loads(manifest_path.read_text())
            for path in manifest_path.parent.glob('.providers/sarvam/*/remote.json'):
                state = json.loads(path.read_text())
                token = path.parent.name
                states[token] = state
                recordings[token] = manifest.get('source', {}).get('sha256')
        # Standalone provider users also retain remote state beneath the output root.
        for path in root.glob('**/.providers/sarvam/*/remote.json'):
            states.setdefault(path.parent.name, json.loads(path.read_text()))
        groups = sorted({j['recording'] for j in jobs}) if scope == 'recording' else [scope]
        reports = []
        for group in groups:
            def belongs(token):
                if scope == 'persistent': return True
                if scope == 'invocation': return invocation in details.get(token, {}).get('invocations', [details.get(token, {}).get('invocation')])
                return details.get(token, {}).get('recording', recordings.get(token)) == group
            selected = [j for j in jobs if scope != 'recording' or j['recording'] == group]
            committed = reserved = 0.0
            for token, cost in reservations.items():
                if not belongs(token): continue
                state = states.get(token, {})
                if state.get('phase') in {'starting', 'started', 'downloaded'} or state.get('remote_state') in {'Completed', 'Failed', 'PartiallyCompleted'}:
                    committed += cost
                else:
                    reserved += cost
            pending = []
            for job in selected:
                old = reservations.get(job['token'], 0)
                # Recovery is free of additional local reservations. Existing remote
                # IDs and ambiguous creates must never turn into replacement jobs.
                increment = 0.0 if job.get('recoverable') else max(0.0, job['cost'] - old)
                pending.append(dict(job, additional_reservation_inr=round(increment, 6)))
            additional = sum(j['additional_reservation_inr'] for j in pending)
            # A previously reserved but not submitted token is still pending in a
            # new invocation, and must consume its selected scope's allowance.
            scoped_existing = sum(reservations.get(j['token'], 0) for j in selected
                                  if not belongs(j['token']) and not j.get('recoverable'))
            required = committed + reserved + additional + scoped_existing
            permitted = not any(not j.get('recoverable') for j in selected) or (c.sarvam_budget > 0 and required <= c.sarvam_budget + 1e-9)
            reports.append({'recording': group if scope == 'recording' else None,
                'configured_budget_inr': c.sarvam_budget, 'scope': scope,
                'previous_committed_cost_inr': round(committed, 6),
                'outstanding_reserved_cost_inr': round(reserved, 6), 'pending_modes': pending,
                'additional_reservation_inr': round(additional + scoped_existing, 6),
                'remaining_budget_inr': round(c.sarvam_budget - committed - reserved, 6),
                'remaining_after_plan_inr': round(c.sarvam_budget - required, 6),
                'processing_permitted': permitted,
                'blocked_reason': None if permitted else 'Local application budget exceeded; Sarvam account credits were not queried'})
        return {'scope': scope, 'currency': 'INR', 'account_credits_checked': False,
                'processing_permitted': all(r['processing_permitted'] for r in reports), 'breakdown': reports}

    if not reserve:
        data = json.loads(ledger.read_text()) if ledger.exists() else {'reservations': {}}
        return calculate(data)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.sarvam-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(ledger.read_text()) if ledger.exists() else {'currency': 'INR', 'reservations': {}}
        report = calculate(data)
        if not report['processing_permitted']:
            raise BudgetPreflightError('Cumulative/local Sarvam budget preflight blocked; no new job created. ' + json.dumps(report))
        for job in jobs:
            if job.get('recoverable'): continue
            data['reservations'][job['token']] = max(data['reservations'].get(job['token'], 0), job['cost'])
            entry = data.setdefault('entries', {}).setdefault(job['token'], {})
            runs = entry.setdefault('invocations', [entry['invocation']] if entry.get('invocation') else [])
            if invocation not in runs: runs.append(invocation)
            entry.update(recording=job['recording'], mode=job['mode'])
        atomic_json(ledger, data)
        return report


def reserve(c, token, cost):
    reserve_many(c, {token: cost})


def reserve_many(c, costs):
    # Per-stage calls are idempotent after the whole-plan reservation. Keep the
    # legacy entry point for independent provider callers and older integrations.
    ledger = Path(c.output_dir) / '.sarvam-budget.json'
    existing = json.loads(ledger.read_text()).get('reservations', {}) if ledger.exists() else {}
    if all(existing.get(t, 0) >= cost for t, cost in costs.items()):
        return
    jobs = [{'token': t, 'cost': cost, 'recording': 'legacy', 'mode': 'unknown'} for t, cost in costs.items()]
    budget_preflight(c, jobs, reserve=True)


def normalize_response(response, mapping, *, diarization=False, mode='codemix', timestamp_policy='strict', speaker_names=None):
    if not isinstance(response, dict):
        raise PipelineError('Saaras response must be a JSON object; original response retained')
    diarized = response.get('diarized_transcript') or {}
    if not isinstance(diarized, dict):
        raise PipelineError('Invalid Saaras diarization object; raw response retained')
    entries = diarized.get('entries') if diarization else None
    if diarization and not entries:
        raise PipelineError('Saaras response is missing requested diarization; raw response retained')
    text = response.get('transcript')
    if not isinstance(text, str) or not text.strip():
        raise PipelineError('Saaras returned no full transcript; raw response retained')
    if entries is None:
        ts = response.get('timestamps') or {}
        if not isinstance(ts, dict):
            ts = {}
        chunks = ts.get('chunks', ts.get('words', []))
        starts, ends = ts.get('start_time_seconds', []), ts.get('end_time_seconds', [])
        if (not all(isinstance(value, list) for value in (chunks, starts, ends))
                or not chunks or not len(chunks) == len(starts) == len(ends)):
            if timestamp_policy == 'strict':
                raise PipelineError('Saaras response has missing or misaligned chunk timestamps; raw response retained')
            entries = [{'transcript': text, 'start_time_seconds': None, 'end_time_seconds': None}]
        else:
            entries = [{'transcript': chunk, 'start_time_seconds': start, 'end_time_seconds': end}
                       for chunk, start, end in zip(chunks, starts, ends)]
    if not isinstance(entries, list) or not entries:
        raise PipelineError('Saaras response has no transcript chunks; raw response retained')
    segments = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PipelineError('Invalid Saaras diarization entry; raw response retained')
        text = entry.get('transcript')
        if not isinstance(text, str):
            raise PipelineError('Invalid Saaras chunk text; raw response retained')
        start, end = entry.get('start_time_seconds'), entry.get('end_time_seconds')
        mapping_error = None
        try:
            if type(start) not in (int, float) or type(end) not in (int, float):
                raise PipelineError('Invalid Saaras chunk timestamps; raw response retained')
            a, b, spans = map_segment(start, end, mapping)
        except PipelineError as e:
            if timestamp_policy == 'strict':
                raise
            a, b, spans, mapping_error = None, None, [], str(e)
        segment = {'id': f's{index:06d}', 'text': text, 'start': a, 'end': b,
                   'processed_start': start, 'processed_end': end, 'original_spans': spans}
        if diarization:
            speaker_id = entry.get('speaker_id')
            segment['speaker_id'] = speaker_id  # Preserve exactly; labels/names never replace provider IDs.
            valid = type(speaker_id) in (str, int) and not any(ord(ch) < 32 for ch in str(speaker_id)) and str(speaker_id).strip().lower() not in {'', '-1', 'unknown', 'none', 'null', 'unidentified', 'uncertain'}
            uncertain = entry.get('speaker_uncertain') is True or entry.get('speaker_confidence') == 0
            status = 'uncertain' if uncertain else 'identified' if valid else 'unknown'
            segment['speaker_status'] = status
            if 'speaker_confidence' in entry: segment['speaker_confidence'] = entry['speaker_confidence']
            name = (speaker_names or {}).get(str(speaker_id)) if valid and not uncertain else None
            segment['speaker_name'] = name
            segment['speaker_label'] = name or (f'{mode} speaker {speaker_id}' if valid and not uncertain else f'{mode} {status} speaker')
        elif 'speaker_id' in entry:
            segment['speaker_id'] = entry['speaker_id']
        if mapping_error:
            segment['mapping_error'] = mapping_error
        segments.append(segment)
    text = response.get('transcript')
    if not isinstance(text, str) or not text.strip():
        raise PipelineError('Saaras returned no full transcript; raw response retained')
    return {'text': text, 'segments': segments, 'language': 'en' if mode == 'translate' else response.get('language_code'),
            'source_language_code': response.get('language_code'),
            'provider': 'sarvam', 'model': 'saaras:v4', 'mode': mode,
            'timestamp_basis': 'original_recording', 'diarization': diarization,
            'speaker_namespace': mode if diarization else None,
            'request_id': response.get('request_id')}


def job_token(src, settings, preprocessing):
    # Output rendering and timestamp validation policy do not change paid inference.
    paid_settings = {key: value for key, value in settings.items()
                     if key not in {'formats', 'filename', 'timestamp_policy', 'speaker_names'}}
    return fingerprint({'source': src, 'settings': paid_settings, 'audio': preprocessing['audio_sha256']})


class SarvamRecognition:
    name = 'sarvam'

    def recognize(self, audio, preprocessing, c, directory, src, settings, record, save, *, transport=None, before_upload=None):
        key_env = getattr(c, 'sarvam_key_env', 'SARVAM_API_KEY')
        key = os.environ.get(key_env)
        mode = settings.get('task', 'codemix')
        model = settings.get('model', c.model)
        if mode not in {'codemix', 'translate'} or model != 'saaras:v4':
            raise PipelineError('Saaras Batch requires saaras:v4 with codemix or translate mode')
        response_file = (getattr(c, 'sarvam_translation_response_file', '') if mode == 'translate'
                         else getattr(c, 'sarvam_response_file', ''))
        resume_job = (getattr(c, 'sarvam_translation_job_id', '') if mode == 'translate' else c.sarvam_job_id)
        resume_flag = '--sarvam-translation-job-id' if mode == 'translate' else '--sarvam-job-id'
        # A local/raw result may be reused without credentials or permission for another upload.
        token = job_token(src, settings, preprocessing)
        work = Path(directory) / '.providers' / 'sarvam' / token
        if not work.resolve().is_relative_to(Path(directory).resolve()):
            raise PipelineError('Provider state directory escapes output directory')
        work.mkdir(parents=True, exist_ok=True)
        state_path, raw_path = work / 'remote.json', work / 'provider-response.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'token': token, 'phase': 'new'}
        cost = estimate(preprocessing['retained_duration'], c)

        def persist():
            atomic_json(state_path, state)
            record['provider'] = {'backend': 'sarvam', 'model': model, 'mode': mode,
                                  'remote_job_id': state.get('job_id'), 'phase': state['phase'],
                                  'state_path': str(state_path.relative_to(directory)), 'cost': cost,
                                  'diarization': c.sarvam_diarization, 'num_speakers': getattr(c, 'sarvam_num_speakers', None),
                                  'speaker_id_scope': 'independent provider job',
                                  'processed_audio_sha256': preprocessing['audio_sha256'],
                                  'uploaded_audio_duration': (preprocessing['retained_duration'] if state['phase'] in
                                       {'uploaded', 'starting', 'started', 'downloaded'} else 0.0),
                                  'processed_audio_duration': preprocessing['retained_duration'],
                                  'audio_codec': preprocessing.get('codec', audio.suffix[1:]),
                                  'audio_size_bytes': audio.stat().st_size, 'remote_status': state.get('remote_state')}
            if state.get('origin') == 'user_supplied_response':
                record['provider'].update(origin='user_supplied_response', model_version_verified=False,
                    cost={'currency': 'INR', 'estimated_cost': 0.0, 'note': 'Existing response imported; no provider call'})
            save()

        def converted(raw):
            result = normalize_response(raw, preprocessing['mapping'], diarization=c.sarvam_diarization, mode=mode,
                                        timestamp_policy=getattr(c, 'sarvam_timestamp_policy', 'strict'),
                                        speaker_names=getattr(c, 'sarvam_speaker_names', {}).get(mode, {}))
            if c.sarvam_diarization:
                result['speaker_namespace'] = {'mode': mode, 'job_id': state.get('job_id'), 'scope': 'independent provider job'}
            record['provider']['raw_response'] = str(raw_path.relative_to(directory))
            record['provider']['raw_response_sha256'] = sha256(raw_path)
            save()
            return result

        if raw_path.is_file() and state.get('response_sha256') == sha256(raw_path):
            persist()
            return converted(json.loads(raw_path.read_bytes()))
        if response_file:
            if src['sha256'] != c.sarvam_response_source_sha256:
                raise PipelineError('Imported Saaras response does not match the configured source SHA-256; no provider call made')
            atomic_bytes(raw_path, Path(response_file).read_bytes())
            state.update(phase='imported', response_sha256=sha256(raw_path), origin='user_supplied_response',
                         model_version_verified=False)
            persist()
            record['provider'].update(origin='user_supplied_response', model_version_verified=False,
                                      cost={'currency': 'INR', 'estimated_cost': 0.0, 'note': 'Existing response imported; no provider call'})
            save()
            return converted(json.loads(raw_path.read_bytes()))
        if getattr(c, 'offline', False):
            raise PipelineError('Sarvam result is not cached and runtime.offline=true; no remote request made')
        if not key:
            raise PipelineError(f'{key_env} is not set; no audio uploaded. Inject it through op run or your environment, never TOML')
        if preprocessing['retained_duration'] > 7200:
            per_mode = math.ceil(preprocessing['retained_duration'] / 7200)
            modes = int(bool(getattr(c, 'transcript', True) and getattr(c, 'recognition_backend', 'sarvam') == 'sarvam')) + int(bool(getattr(c, 'translation', False) and getattr(c, 'translation_backend', '') == 'sarvam'))
            raise PipelineError(f'Saaras Batch limit is 2 hours per file. Splitting would require {per_mode} jobs per mode '
                                f'({per_mode * modes} total selected jobs). No audio uploaded; automatic splitting is disabled. '
                                'Explicitly choose shorter input recordings or revise silence removal before authorizing additional charges')
        if state['phase'] == 'new' and not c.upload_authorized:
            raise PipelineError('Audio upload requires explicit --allow-upload for this invocation; review --dry-run first')
        if (resume_job and state.get('job_id') != resume_job
                and not (state['phase'] == 'creating' and not state.get('job_id'))):
            raise PipelineError(f'{resume_flag} can only reconcile a saved ambiguous create, or match the existing job; refusing to create a replacement')
        if state['phase'] == 'creating' and not state.get('job_id'):
            if not resume_job:
                raise PipelineError(f'Previous create request has an unknown outcome. Reconcile in Sarvam dashboard and supply {resume_flag}; refusing a duplicate job')
            state.update(job_id=resume_job, phase='created')
            persist()
        client = transport or Transport(key, c.request_timeout, getattr(c, 'sarvam_read_retries', 2))
        if state['phase'] == 'new':
            if before_upload:
                before_upload()
            reserve(c, token, cost['estimated_cost'])
            state.update(phase='creating', upload_consent=True, audio_sha256=preprocessing['audio_sha256'])
            persist()  # Before any potentially ambiguous mutation.
            language = 'unknown' if c.language in {'auto', 'unknown'} else (c.language if '-' in c.language else c.language + '-IN')
            params = {'model': model, 'mode': mode, 'language_code': language,
                      'keyterms': c.vocabulary, 'with_timestamps': getattr(c, 'sarvam_timestamps', True), 'with_diarization': c.sarvam_diarization}
            if getattr(c, 'sarvam_num_speakers', None) is not None:
                params['num_speakers'] = c.sarvam_num_speakers
            try:
                response = client.api('POST', payload={'job_parameters': params})
            except SarvamHTTPError as e:
                if 400 <= e.status < 500:
                    state.update(phase='new', last_http_status=e.status)
                    persist()
                raise
            job_id = response.get('job_id')
            if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', job_id):
                raise PipelineError('Sarvam create returned no valid job ID; reconcile before retrying')
            state.update(job_id=job_id, phase='created')
            persist()
        job_id = state['job_id']
        if state['phase'] == 'created':
            if not state.get('upload_consent') or state.get('audio_sha256') != preprocessing['audio_sha256']:
                raise PipelineError('Saved upload consent does not match this audio; refusing upload')
            filename = 'processed.' + preprocessing.get('upload_extension', upload_format(audio)[0])
            links = client.api('POST', '/upload-files', {'job_id': job_id, 'files': [filename]})
            url = links['upload_urls'][filename]['file_url']
            client.upload(url, audio, links['storage_container_type'])
            state['phase'] = 'uploaded'
            persist()
        if state['phase'] == 'uploaded':
            state['phase'] = 'starting'
            persist()
            try:
                client.api('POST', f'/{job_id}/start')
            except SarvamHTTPError as e:
                if 400 <= e.status < 500:
                    state.update(phase='uploaded', last_http_status=e.status)
                    persist()
                raise
            state['phase'] = 'started'
            persist()
        deadline = time.monotonic() + c.sarvam_poll_timeout
        while True:
            status = client.api('GET', f'/{job_id}/status')
            remote = status.get('job_state')
            state['remote_state'] = remote
            persist()
            if remote in {'Completed', 'PartiallyCompleted'}:
                break
            if remote == 'Failed':
                raise PipelineError(f'Sarvam job {job_id} failed; no automatic replacement/billable retry')
            if state['phase'] == 'starting' and remote == 'Accepted':
                raise PipelineError('Start outcome is uncertain; inspect/start the existing job in the dashboard. No new job will be created')
            if remote not in {'Accepted', 'Pending', 'Running'}:
                raise PipelineError('Unknown Sarvam job state; job retained for safe resumption')
            if time.monotonic() >= deadline:
                raise PipelineError(f'Sarvam job {job_id} still pending; rerun to poll the same job without re-uploading')
            print(f'Sarvam job {job_id}: {remote}; resumption state saved', flush=True)
            time.sleep(c.sarvam_poll_interval)
        outputs = [out['file_name'] for task in status.get('job_details', []) if task.get('state') == 'Success'
                   for out in task.get('outputs', []) if out['file_name'].endswith('.json')]
        if len(outputs) != 1 or status.get('failed_files_count', 0):
            raise PipelineError('Saaras job has incomplete/ambiguous file outputs; no billable retry performed')
        links = client.api('POST', '/download-files', {'job_id': job_id, 'files': outputs})
        body = client.download(links['download_urls'][outputs[0]]['file_url'])
        # Retain exact provider bytes, even if subsequent parsing/mapping fails.
        atomic_bytes(raw_path, body)
        state.update(phase='downloaded', response_sha256=sha256(raw_path))
        persist()
        return converted(json.loads(body))
