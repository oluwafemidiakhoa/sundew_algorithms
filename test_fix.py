# test_fix.py

import sys

from tests.conftest import get_preset

from src.sundew.core import SundewAlgorithm

# Remove any cached versions of sundew so we import fresh
for name in list(sys.modules):
    if name.startswith("sundew"):
        del sys.modules[name]

cfg = get_preset(
    "tuned_v2",
    overrides={"gate_temperature": 0.15, "energy_pressure": 0.04},
)
algo = SundewAlgorithm(cfg)
stats = algo.report()
print(f"Activated count: {stats.get('activated', 'KEY_MISSING')}")
