from prefect import flow, task
import os, pandas as pd, numpy as np, json
import yfinance as yf
from datetime import datetime, UTC

try:
    import xgboost as xgb
except Exception:
    xgb = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "data")
FEATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "features")
REGISTRY_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "registry")

for d in [DATA_DIR, FEATURE_DIR, REGISTRY_DIR]:
    os.makedirs(d, exist_ok=True)

@task(retries=2, retry_delay_seconds=10)
def ingest(ticker: str, period: str = "2y"):
    df = yf.download(ticker, period=period, progress=False)
    df.reset_index(inplace=True)
    # Flatten potential MultiIndex columns and normalize names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    df.rename(columns={"Date": "date", "Adj Close": "Adj_Close"}, inplace=True)
    df["ticker"] = ticker
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    df.to_parquet(path)
    return path

@task
def engineer(data_path: str):
    df = pd.read_parquet(data_path).sort_values("date")
    close = df["Close"]
    df["return"] = close.pct_change()
    df["log_return"] = np.log(close).diff()
    # Indicators
    for w in [5, 10, 20]:
        df[f"sma_{w}"] = close.rolling(w).mean()
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
    df["vol_10"] = df["return"].rolling(10).std()
    # RSI 14
    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    # Lags
    for lag in [1, 2, 3, 5]:
        df[f"lag_close_{lag}"] = close.shift(lag)
        df[f"lag_return_{lag}"] = df["return"].shift(lag)
    df.dropna(inplace=True)
    fpath = os.path.join(FEATURE_DIR, os.path.basename(data_path))
    df.to_parquet(fpath)
    return fpath

@task
def train(feature_path: str):
    df = pd.read_parquet(feature_path)
    # next-day return target
    y = df["return"].shift(-1)
    feature_cols = [c for c in df.columns if c not in ["date", "ticker", "return"] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols]
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[mask]
    y = y.loc[mask]

    # Baseline weights for fallback
    weights = np.ones(X.shape[1]) / X.shape[1]
    base_meta = {
        "name": "prefect-feature-baseline",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_count": int(X.shape[1]),
        "weights": weights.tolist(),
        "features": feature_cols,
        "notes": "Placeholder baseline; XGBoost may overwrite"
    }
    with open(os.path.join(REGISTRY_DIR, "baseline_model.json"), "w") as f:
        json.dump(base_meta, f)

    # Train XGBoost if available
    if xgb is None:
        return os.path.join(REGISTRY_DIR, "baseline_model.json")

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
    rmse = float(np.sqrt(((y_test - pred) ** 2).mean()))
    mae = float(np.abs(y_test - pred).mean())
    # Save booster and metadata with clean feature names
    booster = model.get_booster()
    booster.save_model(os.path.join(REGISTRY_DIR, "xgb_model.ubj"))
    meta = {
        "name": "xgb-regressor",
        "created_at": datetime.now(UTC).isoformat(),
        "features": list(X.columns),
        "metrics": {"rmse": rmse, "mae": mae},
        "artifact": "xgb_model.ubj",
    }
    with open(os.path.join(REGISTRY_DIR, "xgb_model.json"), "w") as f:
        json.dump(meta, f)
    return os.path.join(REGISTRY_DIR, "xgb_model.json")

@flow
def pipeline(ticker: str = "AAPL"):
    p = ingest(ticker)
    f = engineer(p)
    m = train(f)
    return m

if __name__ == "__main__":
    pipeline()
