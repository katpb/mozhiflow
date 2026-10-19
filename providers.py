"""Compatibility import for :mod:`mozhiflow.providers`."""
import sys
from mozhiflow import providers as _implementation

sys.modules[__name__] = _implementation
