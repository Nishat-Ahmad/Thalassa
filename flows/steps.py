from prefect import task
import os, pandas as pd, numpy as np, json
import yfinance as yf
from datetime import datetime, UTC
from subprocess import Popen, PIPE
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:
    PCA = None
    KMeans = None
    StandardScaler = None

try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "data")
FEATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "features")
REGISTRY_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "registry")

for d in [DATA_DIR, FEATURE_DIR, REGISTRY_DIR]:
    os.makedirs(d, exist_ok=True)


def _ticker_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base.upper()

@task(retries=2, retry_delay_seconds=10)
def ingest(ticker: str, period: str = "2y"):
    df = yf.download(ticker, period=period, progress=False)
    df.reset_index(inplace=True)
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
    df.dropna(inplace=True)
    fpath = os.path.join(FEATURE_DIR, os.path.basename(data_path))
    df.to_parquet(fpath)
    return fpath

@task
def train_regressor(feature_path: str, run_dir: str | None = None):
    df = pd.read_parquet(feature_path)
    y = df["return"].shift(-1)
    feature_cols = [c for c in df.columns if c not in ["date", "ticker", "return"] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols]
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[mask]
    y = y.loc[mask]
    if xgb is None:
        raise RuntimeError("xgboost not installed; cannot train regressor")
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
    booster = model.get_booster()
    ticker = _ticker_from_path(feature_path)
    target_dir = run_dir or os.path.join(REGISTRY_DIR, ticker)
    os.makedirs(target_dir, exist_ok=True)
    model_path = os.path.join(target_dir, f"xgb_model_{ticker}.ubj")
    meta_path = os.path.join(target_dir, f"xgb_model_{ticker}.json")
    booster.save_model(model_path)
    meta = {
        "name": "xgb-regressor",
        "created_at": datetime.now(UTC).isoformat(),
        "features": list(X.columns),
        "metrics": {"rmse": rmse, "mae": mae},
        "artifact": os.path.basename(model_path),
        "ticker": ticker,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta_path

@task
def train_classification(feature_path: str, run_dir: str | None = None):
    ticker = _ticker_from_path(feature_path)
    script = os.path.join(os.path.dirname(__file__), "..", "ml", "train_classification.py")
    cmd = ["python", script, "--ticker", ticker]
    if run_dir:
        cmd.extend(["--run-dir", run_dir])
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Classification training failed: {stderr.decode()}")
    try:
        result = json.loads(stdout.decode())
    except Exception:
        result = {"status": "success"}
    return result

@task
def train_association_rules(
    feature_path: str,
    run_dir: str | None = None,
    min_support: float = 0.1,
    min_confidence: float = 0.6,
    max_rules: int = 100,
):
    ticker = _ticker_from_path(feature_path)
    script = os.path.join(os.path.dirname(__file__), "..", "ml", "train_association.py")
    target_dir = run_dir or os.path.join(REGISTRY_DIR, ticker)
    cmd = [
        "python",
        script,
        "--features",
        feature_path,
        "--registry",
        target_dir,
        "--min-support",
        str(min_support),
        "--min-confidence",
        str(min_confidence),
        "--max-rules",
        str(max_rules),
    ]
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Association training failed: {stderr.decode()}")
    try:
        result = json.loads(stdout.decode())
    except Exception:
        result = {"status": "success"}
    return result

@task
def compute_pca(feature_path: str, n_components: int = 5, run_dir: str | None = None):
    if PCA is None:
        return {"status": "skipped", "reason": "sklearn not available"}
    df = pd.read_parquet(feature_path)
    feature_cols = [c for c in df.columns if c not in ["date", "ticker"] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        return {"status": "skipped", "reason": "no data"}
    pca = PCA(n_components=min(n_components, X.shape[1]))
    comps = pca.fit_transform(X)
    try:
        dates = pd.to_datetime(df.loc[X.index, "date"]).dt.strftime("%Y-%m-%d").tolist()
    except Exception:
        dates = [str(i) for i in X.index.tolist()]
    ticker = _ticker_from_path(feature_path)
    target_dir = run_dir or os.path.join(REGISTRY_DIR, ticker)
    os.makedirs(target_dir, exist_ok=True)
    meta = {
        "name": "pca_features",
        "created_at": datetime.now(UTC).isoformat(),
        "n_components": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "feature_order": feature_cols,
        "row_index": dates,
        "ticker": ticker,
    }
    with open(os.path.join(target_dir, f"pca_{ticker}.json"), "w") as f:
        json.dump(meta, f)
    np.save(os.path.join(target_dir, f"pca_transformed_{ticker}.npy"), comps)
    return {"status": "ok", "pca_meta": meta}

@task
def cluster_features(feature_path: str, n_clusters: int = 5, run_dir: str | None = None):
    if KMeans is None:
        return {"status": "skipped", "reason": "sklearn not available"}
    df = pd.read_parquet(feature_path)
    feature_cols = [c for c in df.columns if c not in ["date", "ticker"] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < n_clusters:
        return {"status": "skipped", "reason": "insufficient rows"}
    scaler = StandardScaler() if StandardScaler else None
    X_scaled = scaler.fit_transform(X) if scaler else X.to_numpy()
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X_scaled)
    ticker = _ticker_from_path(feature_path)
    target_dir = run_dir or os.path.join(REGISTRY_DIR, ticker)
    os.makedirs(target_dir, exist_ok=True)
    meta = {
        "name": "kmeans_clusters",
        "created_at": datetime.now(UTC).isoformat(),
        "n_clusters": n_clusters,
        "inertia": float(km.inertia_),
        "centers": km.cluster_centers_.tolist(),
        "feature_order": feature_cols,
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "ticker": ticker,
    }
    if scaler is not None:
        try:
            meta["scaler_mean"] = scaler.mean_.tolist()
            meta["scaler_scale"] = scaler.scale_.tolist()
        except Exception:
            pass
    with open(os.path.join(target_dir, f"clusters_{ticker}.json"), "w") as f:
        json.dump(meta, f)
    np.save(os.path.join(target_dir, f"cluster_labels_{ticker}.npy"), labels)
    return {"status": "ok", "cluster_meta": meta}

@task
def forecast_ts(feature_path: str, horizon: int = 7, run_dir: str | None = None):
    if ARIMA is None:
        return {"status": "skipped", "reason": "statsmodels not available"}
    df = pd.read_parquet(feature_path).sort_values("date")
    if "Close" not in df.columns:
        return {"status": "skipped", "reason": "Close column missing"}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    series = df.set_index("date")["Close"].astype(float).sort_index()
    freq = pd.infer_freq(series.index)
    if freq is None:
        freq = "D"
    series = series.asfreq(freq).ffill()
    order = (1, 1, 1)
    try:
        model = ARIMA(series, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        conf_res = fitted.get_forecast(steps=horizon)
        conf = conf_res.conf_int()
    except Exception as e:
        return {"status": "error", "message": str(e)}
    last_date = series.index[-1]
    idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq=freq or "D")
    ticker = _ticker_from_path(feature_path)
    target_dir = run_dir or os.path.join(REGISTRY_DIR, ticker)
    os.makedirs(target_dir, exist_ok=True)
    out = {
        "name": "arima-forecast",
        "created_at": datetime.now(UTC).isoformat(),
        "horizon": int(horizon),
        "order": list(order),
        "aic": float(getattr(fitted, 'aic', float('nan'))),
        "bic": float(getattr(fitted, 'bic', float('nan'))),
        "last_observation": float(series.iloc[-1]),
        "predictions": [float(x) for x in forecast.tolist()],
        "dates": [d.isoformat() for d in idx],
        "confidence_interval": conf.values.tolist(),
        "ticker": ticker,
    }
    with open(os.path.join(target_dir, f"forecast_{ticker}.json"), "w") as f:
        json.dump(out, f)
    return {"status": "ok", "forecast": out}
