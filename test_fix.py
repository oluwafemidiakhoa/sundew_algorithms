# test_fix.py
import sys

from src.sundew.core import SundewAlgorithm
from tests.conftest import get_preset

# Remove any cached versions
modules_to_remove = [k for k in list(sys.modules.keys()) if "sundew" in k]
for m in modules_to_remove:
    del sys.modules[m]

cfg = get_preset("tuned_v2", overrides=dict(gate_temperature=0.15, energy_pressure=0.04))
algo = SundewAlgorithm(cfg)
stats = algo.report()
print(f"Activated count: {stats.get('activated', 'KEY_MISSING')}")
