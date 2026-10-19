"""Compatibility import for :mod:`mozhiflow.pipeline`."""
import sys
from mozhiflow import pipeline as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.worker_main() if sys.argv[1:] == ['--_worker'] else _implementation.main())

sys.modules[__name__] = _implementation
