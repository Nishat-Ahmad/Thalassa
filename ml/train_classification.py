import os
import json
import datetime as dt
import numpy as np
import pandas as pd
import argparse

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


def _paths_for_ticker(ticker: str, registry_dir: str | None = None) -> tuple[str, str]:
    base = registry_dir or REGISTRY_DIR
    t = ticker.upper()
    model_path = os.path.join(base, f"xgb_classifier_{t}.ubj")
    meta_path = os.path.join(base, f"xgb_classifier_{t}.json")
    return model_path, meta_path


def load_latest_features(ticker: str) -> pd.DataFrame:
    if not os.path.isdir(FEATURES_DIR):
        os.makedirs(FEATURES_DIR, exist_ok=True)
        # Attempt to bootstrap features if none are present
        bootstrap_features(ticker=ticker)
    files = [
        f
        for f in os.listdir(FEATURES_DIR)
        if f.endswith(".parquet") or f.endswith(".csv")
    ]
    if not files:
        bootstrap_features(ticker=ticker)
        files = [
            f
            for f in os.listdir(FEATURES_DIR)
            if f.endswith(".parquet") or f.endswith(".csv")
        ]
        if not files:
            raise FileNotFoundError("no feature files found after bootstrap")
    # Prefer file that matches ticker
    target_file = None
    ticker_upper = ticker.upper()
    for f in sorted(files):
        name = os.path.splitext(f)[0].upper()
        if name == ticker_upper:
            target_file = f
    if target_file is None:
        files.sort()
        target_file = files[-1]
    path = os.path.join(FEATURES_DIR, target_file)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # Flatten multiindex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x) for x in tup if str(x) != ""]) for tup in df.columns
        ]
    return df


def bootstrap_features(ticker: str = "AAPL", period: str = "2y"):
    """Minimal feature generation to unblock training if Prefect flow hasn't run."""
    if yf is None:
        return
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
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


def build_labels(
    df: pd.DataFrame, target_col: str | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    # Use next-day return classification: up (1) if return > 0 else down (0)
    # If a `log_return` exists, use it; else compute from Close.
    dfx = df.copy()
    # prefer an explicit target_col if provided
    if target_col and target_col in dfx.columns:
        ret = dfx[target_col]
    else:
        # compute next-day return from Close whenever possible (avoids using
        # current-day 'return' / 'log_return' which would leak the target)
        if "Close" in dfx.columns:
            close = dfx["Close"].astype(float)
            ret = (close.shift(-1) - close) / close
        elif "log_return" in dfx.columns:
            # if only log_return present, assume it's current-day and shift it
            ret = dfx["log_return"].shift(-1)
        else:
            raise ValueError("No suitable target columns found (Close or log_return)")

    y = (ret > 0).astype(int).to_numpy()
    # Drop rows with NaN target
    valid = ~np.isnan(ret)
    dfx = dfx.loc[valid]
    y = y[valid.values if hasattr(valid, "values") else valid]
    # Keep only numeric features
    X = dfx.select_dtypes(include=[np.number])
    # Drop columns that directly encode the target (current-day return/log_return)
    for col in ["return", "log_return"]:
        if col in X.columns:
            X = X.drop(columns=[col])
    # Replace infinities with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    # Impute missing feature values: forward/backward fill then median
    try:
        X = X.ffill().bfill()
    except Exception:
        pass
    for c in X.columns:
        if X[c].isna().any():
            try:
                med = float(X[c].median(skipna=True))
                if np.isnan(med):
                    med = 0.0
            except Exception:
                med = 0.0
            X[c] = X[c].fillna(med)
    # Align y
    y = y[: len(X)]
    return X, y


def train_classifier(ticker: str = "AAPL", registry_dir: str | None = None):
    if xgb is None:
        raise RuntimeError("xgboost not available")
    df = load_latest_features(ticker)
    X, y = build_labels(df)
    # Minimum samples required to attempt training. Lowered to be more permissive
    MIN_SAMPLES = 30
    if len(X) < MIN_SAMPLES:
        # return a skipped result instead of raising so the pipeline can continue
        return {
            "status": "skipped",
            "reason": "not enough samples to train classifier",
            "samples": int(len(X)),
        }

    # Split into train / validation so we evaluate generalization (avoid
    # reporting optimistic train-only metrics)
    try:
        from sklearn.model_selection import train_test_split

        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except Exception:
        # fallback: simple contiguous split if sklearn not available
        split = max(int(len(X) * 0.8), 1)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y[: split], y[split:]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=[str(c) for c in X.columns])
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=[str(c) for c in X.columns])
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    # Use early stopping on validation set to reduce overfitting
    watchlist = [(dtrain, "train"), (dval, "validation")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=watchlist,
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    # Predictions on train and validation
    preds_train = booster.predict(dtrain)
    preds_val = booster.predict(dval)

    # Metrics
    eps = 1e-15
    logloss_train = float(
        np.mean(-(y_train * np.log(preds_train + eps) + (1 - y_train) * np.log(1 - preds_train + eps)))
    )
    logloss_val = float(
        np.mean(-(y_val * np.log(preds_val + eps) + (1 - y_val) * np.log(1 - preds_val + eps)))
    )
    auc_train = float(np.nan)
    auc_val = float(np.nan)
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_train)) > 1:
            auc_train = float(roc_auc_score(y_train, preds_train))
        if len(np.unique(y_val)) > 1:
            auc_val = float(roc_auc_score(y_val, preds_val))
    except Exception:
        pass

    # Calibration / reliability: Brier score + calibration curve on validation
    calibration = {}
    try:
        from sklearn.metrics import brier_score_loss
        from sklearn.calibration import calibration_curve

        brier = float(brier_score_loss(y_val, preds_val))
        prob_true, prob_pred = calibration_curve(y_val, preds_val, n_bins=10, strategy="uniform")
        calibration = {
            "brier": brier,
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
    except Exception:
        calibration = {}

    registry = registry_dir or REGISTRY_DIR
    os.makedirs(registry, exist_ok=True)
    model_path, meta_path = _paths_for_ticker(ticker, registry)
    booster.save_model(model_path)
    meta = {
        "task": "classification",
        "model": "xgb_classifier",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "features": [str(c) for c in X.columns],
        "metrics": {
            "logloss_train": float(logloss_train),
            "auc_train": float(auc_train),
            "logloss_val": float(logloss_val),
            "auc_val": float(auc_val),
        },
        "calibration": calibration,
        "samples": {"total": int(len(X)), "train": int(len(y_train)), "val": int(len(y_val))},
        "ticker": ticker.upper(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {
        "status": "success",
        "model_path": model_path,
        "meta_path": meta_path,
        "metrics": meta["metrics"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGB classifier")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to train on")
    parser.add_argument("--registry", default=None, help="Override registry directory")
    parser.add_argument(
        "--run-dir", default=None, help="Optional run directory to place outputs"
    )
    args = parser.parse_args()
    registry = args.run_dir or args.registry
    out = train_classifier(args.ticker, registry_dir=registry)
    print(json.dumps(out, indent=2))
