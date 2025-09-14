from sundew import SundewAlgorithm
from sundew.config import SundewConfig

# Configure (same spirit as your demo)
cfg = SundewConfig(
    activation_threshold=0.78,
    target_activation_rate=0.15,
    gate_temperature=0.08,
    max_threshold=0.92,
    energy_pressure=0.04,
)

algo = SundewAlgorithm(cfg)

sample = {
    "type": "sensor",
    "magnitude": 55,          # 0–100 scale (we normalize)
    "anomaly_score": 0.30,    # 0–1
    "context_relevance": 0.5,  # 0–1
    "urgency": 0.1,           # 0–1
}

res = algo.process(sample)
if res is None:
    print("Dormant (not processed)")
else:
    print(f"Activated: significance={res.significance:.3f}, energy={res.energy_consumed:.2f}")
