import os
import json
import math
import datetime as dt
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import yfinance as yf
except Exception:
    yf = None

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
# Align with Prefect flow: features saved under ml/features
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "features")
REGISTRY_DIR = os.path.join(BASE_DIR, "registry")

CLS_MODEL_PATH = os.path.join(REGISTRY_DIR, "xgb_classifier.ubj")
CLS_META_PATH = os.path.join(REGISTRY_DIR, "xgb_classifier.json")


def load_latest_features() -> pd.DataFrame:
    if not os.path.isdir(FEATURES_DIR):
        os.makedirs(FEATURES_DIR, exist_ok=True)
        # Attempt to bootstrap features if none are present
        bootstrap_features()
    files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.parquet') or f.endswith('.csv')]
    if not files:
        bootstrap_features()
        files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.parquet') or f.endswith('.csv')]
        if not files:
            raise FileNotFoundError("no feature files found after bootstrap")
    files.sort()
    path = os.path.join(FEATURES_DIR, files[-1])
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # Flatten multiindex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join([str(x) for x in tup if str(x) != '']) for tup in df.columns]
    return df


def bootstrap_features(ticker: str = "AAPL", period: str = "2y"):
    """Minimal feature generation to unblock training if Prefect flow hasn't run."""
    if yf is None:
        return
    df = yf.download(ticker, period=period, progress=False)
    if df is None or df.empty:
        return
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    df.rename(columns={"Date": "date", "Adj Close": "Adj_Close"}, inplace=True)
    df["ticker"] = ticker
    close = df["Close"].astype(float)
    df["return"] = close.pct_change()
    df["log_return"] = np.log(close).diff()
    for w in [5, 10, 20]:
        df[f"sma_{w}"] = close.rolling(w).mean()
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
    df["vol_10"] = df["return"].rolling(10).std()
    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    for lag in [1, 2, 3, 5]:
        df[f"lag_close_{lag}"] = close.shift(lag)
        df[f"lag_return_{lag}"] = df["return"].shift(lag)
    df = df.dropna()
    out = os.path.join(FEATURES_DIR, f"{ticker}.parquet")
    df.to_parquet(out)


def build_labels(df: pd.DataFrame, target_col: str | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    # Use next-day return classification: up (1) if return > 0 else down (0)
    # If a `log_return` exists, use it; else compute from Close.
    dfx = df.copy()
    if target_col and target_col in dfx.columns:
        ret = dfx[target_col]
    elif 'log_return' in dfx.columns:
        ret = dfx['log_return']
    elif 'Close' in dfx.columns:
        close = dfx['Close'].astype(float)
        ret = (close.shift(-1) - close) / close
    else:
        raise ValueError("No suitable target columns found (log_return or Close)")

    y = (ret > 0).astype(int).to_numpy()
    # Drop rows with NaN target
    valid = ~np.isnan(ret)
    dfx = dfx.loc[valid]
    y = y[valid.values if hasattr(valid, 'values') else valid]
    # Keep only numeric features
    X = dfx.select_dtypes(include=[np.number])
    # Drop target leakage columns
    for col in ['log_return']:
        if col in X.columns:
            X = X.drop(columns=[col])
    # Drop rows with NaNs
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    # Align y
    y = y[: len(X)]
    return X, y


def train_classifier():
    if xgb is None:
        raise RuntimeError("xgboost not available")
    df = load_latest_features()
    X, y = build_labels(df)
    if len(X) < 50:
        raise ValueError("Not enough samples to train classifier")

    dtrain = xgb.DMatrix(X, label=y, feature_names=[str(c) for c in X.columns])
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=200)

    # Simple evaluation on train (for demo); in production, add a holdout split
    preds = booster.predict(dtrain)
    # Metrics
    eps = 1e-15
    logloss = float(np.mean(-(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))))
    auc = float(np.nan)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, preds))
    except Exception:
        pass

    os.makedirs(REGISTRY_DIR, exist_ok=True)
    booster.save_model(CLS_MODEL_PATH)
    meta = {
        "task": "classification",
        "model": "xgb_classifier",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "features": [str(c) for c in X.columns],
        "metrics": {"logloss": logloss, "auc": auc},
        "samples": int(len(X)),
    }
    with open(CLS_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    return {
        "status": "success",
        "model_path": CLS_MODEL_PATH,
        "meta_path": CLS_META_PATH,
        "metrics": meta["metrics"],
    }


if __name__ == "__main__":
    out = train_classifier()
    print(json.dumps(out, indent=2))
