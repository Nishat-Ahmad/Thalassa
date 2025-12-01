import os, json
import numpy as np
import pandas as pd
from datetime import datetime, UTC
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("xgboost not installed. Run: pip install xgboost")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEATURE_DIR = os.path.join(os.path.dirname(__file__), "features")
REGISTRY_DIR = os.path.join(os.path.dirname(__file__), "registry")
os.makedirs(REGISTRY_DIR, exist_ok=True)

FEATURE_FILE = os.path.join(FEATURE_DIR, "AAPL.parquet")
if not os.path.exists(FEATURE_FILE):
    raise SystemExit(f"Feature file not found: {FEATURE_FILE}. Run flows/flow.py first.")

df = pd.read_parquet(FEATURE_FILE)
# Flatten possible MultiIndex and normalize names
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
# Ensure numeric feature matrix only
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in ["return"]]
y = df["return"].shift(-1)
X = df[feature_cols]
mask = (~X.isna().any(axis=1)) & (~y.isna())
X = X.loc[mask]
y = y.loc[mask]

# Train/test split by time
split = int(0.8 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
mae = float(mean_absolute_error(y_test, pred))

artifact_path = os.path.join(REGISTRY_DIR, "xgb_model.json")
model.save_model(os.path.join(REGISTRY_DIR, "xgb_model.ubj"))
meta = {
    "name": "xgb-regressor",
    "created_at": datetime.now(UTC).isoformat(),
    "features": list(X.columns),
    "metrics": {"rmse": rmse, "mae": mae},
    "artifact": "xgb_model.ubj",
}
with open(artifact_path, "w") as f:
    json.dump(meta, f)
print(f"Saved XGB model and metrics to {REGISTRY_DIR}")
