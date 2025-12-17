# tools/check_regressor_baseline.py
import os
import numpy as np
import pandas as pd

TICKER = "SHOP.TO"
feat_path = os.path.join("ml","features", f"{TICKER}.parquet")
if not os.path.exists(feat_path):
    raise SystemExit("features file missing: " + feat_path)
df = pd.read_parquet(feat_path).sort_values("date").reset_index(drop=True)
y = df["return"].shift(-1).dropna()
# align X length
y = y.values
# naive baseline: predict 0 (no-change)
pred0 = np.zeros_like(y)
rmse0 = float(np.sqrt(np.mean((y - pred0)**2)))
mae0 = float(np.mean(np.abs(y - pred0)))
print("Naive baseline (zero) RMSE:", rmse0, "MAE:", mae0)

# If you have saved predictions or want to score model outputs, compute them here.
# Optionally compute standard deviation of y as another reference:
print("Std(next-day return):", float(np.std(y)))