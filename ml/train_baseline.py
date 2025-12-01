import os, json, numpy as np
from datetime import datetime, UTC

REGISTRY_DIR = os.path.join(os.path.dirname(__file__), "registry")
MODEL_PATH = os.path.join(REGISTRY_DIR, "baseline_model.json")

os.makedirs(REGISTRY_DIR, exist_ok=True)
np.random.seed(42)
weights = np.random.uniform(low=-0.5, high=0.5, size=10).tolist()
meta = {
    "name": "baseline-linear",
    "created_at": datetime.now(UTC).isoformat(),
    "weights": weights,
    "metrics": {"note": "demo weights; replace via Prefect-trained model"}
}
with open(MODEL_PATH, "w") as f:
    json.dump(meta, f)
print(f"Saved baseline model to {MODEL_PATH}")
