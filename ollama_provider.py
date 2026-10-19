"""Compatibility import for :mod:`mozhiflow.ollama_provider`."""
import sys
from mozhiflow import ollama_provider as _implementation

sys.modules[__name__] = _implementation
