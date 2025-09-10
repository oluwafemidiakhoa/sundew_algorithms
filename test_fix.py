# test_fix.py

import sys

from sundew.core import SundewAlgorithm
from tests.conftest import get_preset

# Remove any cached versions of the package so we import the freshly installed one.
modules_to_remove = [k for k in list(sys.modules) if k.startswith("sundew")]
for k in modules_to_remove:
    del sys.modules[k]

cfg = get_preset(
    "tuned_v2",
    overrides={"gate_temperature": 0.15, "energy_pressure": 0.04},
)
algo = SundewAlgorithm(cfg)
stats = algo.report()
print(f"Activated count: {stats.get('activated', 'KEY_MISSING')}")
