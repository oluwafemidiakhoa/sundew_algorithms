import json
import os
from pathlib import Path

# Windows-friendly expansion of %USERPROFILE%
p = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\demo_run.json"))

with p.open("r", encoding="utf-8") as f:
    data = json.load(f)

cfg = data.get("config", {})
rep = data.get("report", {})
proc = data.get("processed_events", [])

print("== Sundew demo summary ==")
print(f"file: {p}")
print(f"events              : {rep.get('total_inputs')}")
print(f"activations         : {rep.get('activations')}")
print(f"activation_rate     : {rep.get('activation_rate')}")
print(f"ema_activation_rate : {rep.get('ema_activation_rate'):.3f}")
print(f"threshold(final)    : {rep.get('threshold'):.3f}")
print(f"energy_left         : {rep.get('energy_remaining'):.3f}")
print(f"savings_pct         : {rep.get('estimated_energy_savings_pct'):.2f}%")
print()
print("Top 3 processed events (sig, time[s], ΔE):")
for e in proc[:3]:
    print(f"  {e['significance']:.3f}, {e['processing_time']:.3f}, {e['energy_consumed']:.1f}")
