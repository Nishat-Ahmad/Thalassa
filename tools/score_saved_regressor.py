# tools/score_saved_regressor.py
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

TICKER = "SHOP.TO"
# find latest run dir under ml/registry/TICKER
base = os.path.join("ml", "registry", TICKER)
runs = sorted(os.listdir(base)) if os.path.isdir(base) else []
if not runs:
    raise SystemExit("No runs found")
run = runs[-1]
run_dir = os.path.join(base, run)
meta_path = os.path.join(run_dir, f"xgb_model_{TICKER}.json")
model_path = os.path.join(run_dir, f"xgb_model_{TICKER}.ubj")
feat_path = os.path.join("ml", "features", f"{TICKER}.parquet")

meta = json.load(open(meta_path))
feat_cols = meta.get("features", [])
df = pd.read_parquet(feat_path).sort_values("date").reset_index(drop=True)
y = df["return"].shift(-1)
X = df[feat_cols].loc[y.dropna().index]
y = y.dropna()
dmat = xgb.DMatrix(X)
booster = xgb.Booster()
booster.load_model(model_path)
preds = booster.predict(dmat)
rmse = float(np.sqrt(np.mean((y.values - preds) ** 2)))
mae = float(np.mean(np.abs(y.values - preds)))
print("Saved model RMSE:", rmse, "MAE:", mae)
print("Meta metrics:", meta.get("metrics"))
