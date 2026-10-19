"""Compatibility import for :mod:`mozhiflow.sarvam_provider`."""
import sys
from mozhiflow import sarvam_provider as _implementation

sys.modules[__name__] = _implementation
