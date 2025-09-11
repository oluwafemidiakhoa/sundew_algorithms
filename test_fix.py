# tests/test_fix.py
from __future__ import annotations

import importlib
from typing import Any, Dict

from sundew.config_presets import get_preset
from sundew.core import SundewAlgorithm
from sundew.demo import synth_event


def test_process_smoke() -> None:
    """Simple smoke test: one synthetic event flows through the core."""
    importlib.invalidate_caches()

    algo = SundewAlgorithm(get_preset("tuned_v2"))
    event: Dict[str, Any] = synth_event(temperature=0.1)

    res = algo.process(event)

    # Basic sanity checks
    assert 0.0 <= res.significance <= 1.0
    assert 0.0 <= algo.threshold <= 0.9
