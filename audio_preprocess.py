"""Compatibility import for :mod:`mozhiflow.audio_preprocess`."""
import sys
from mozhiflow import audio_preprocess as _implementation

sys.modules[__name__] = _implementation
