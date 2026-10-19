"""Shared exceptions and small persistence helpers; no provider imports."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


class PipelineError(Exception):
    pass


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.write-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def atomic_bytes(path, value):
    """Replace the directory entry atomically, without following a destination symlink."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.download-')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(value)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
