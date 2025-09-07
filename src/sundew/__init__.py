from __future__ import annotations

from .config import SundewConfig
from .config_presets import get_preset, list_presets
from .core import ProcessingResult, SundewAlgorithm
from .demo import run_demo

__all__ = [
    "SundewAlgorithm",
    "SundewConfig",
    "ProcessingResult",
    "get_preset",
    "list_presets",
    "run_demo",
]
