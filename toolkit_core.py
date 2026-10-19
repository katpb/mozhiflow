"""Compatibility import for :mod:`mozhiflow.toolkit_core`."""
import sys
from mozhiflow import toolkit_core as _implementation

sys.modules[__name__] = _implementation
