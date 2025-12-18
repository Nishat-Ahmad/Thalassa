import os
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen


def _maybe_fix_prefect_api_url_env() -> None:
    """Fix Docker-only PREFECT_API_URL when running on the host.

    Docker Compose uses http://prefect:4200/api. On the host, use localhost.
    This must run before importing Prefect.
    """

    prefect_api_url = os.environ.get("PREFECT_API_URL")
    if not prefect_api_url:
        return

    try:
        parsed = urlparse(prefect_api_url)
    except Exception:
        return

    if parsed.hostname != "prefect":
        return

    def _health_url(api_url: str) -> str:
        return api_url.rstrip("/") + "/health"

    def _is_healthy(api_url: str, timeout_s: float = 1.5) -> bool:
        try:
            with urlopen(_health_url(api_url), timeout=timeout_s) as resp:
                status = getattr(resp, "status", None)
                if status is None:
                    return True
                return 200 <= int(status) < 300
        except Exception:
            return False

    if _is_healthy(prefect_api_url):
        return

    localhost_url = urlunparse(parsed._replace(netloc="localhost:4200"))
    if _is_healthy(localhost_url):
        os.environ["PREFECT_API_URL"] = localhost_url
        return

    # If neither docker-hostname nor localhost is reachable, drop the override so Prefect
    # can fall back to its default API behavior.
    try:
        del os.environ["PREFECT_API_URL"]
    except KeyError:
        pass


_maybe_fix_prefect_api_url_env()

from prefect import task  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import warnings  # noqa: E402
import logging  # noqa: E402
import yfinance as yf  # noqa: E402
from datetime import datetime, UTC  # noqa: E402
from subprocess import Popen, PIPE  # noqa: E402

try:
    from sklearn.decomposition import PCA  # noqa: E402
    from sklearn.cluster import KMeans  # noqa: E402
    from sklearn.preprocessing import StandardScaler  # noqa: E402
except Exception:
    PCA = None
    KMeans = None
    StandardScaler = None

try:
    import xgboost as xgb  # noqa: E402
except Exception:
    xgb = None
try:
    from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
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
def ingest(ticker: str, period: str = "max"):
    logger = logging.getLogger(__name__)
    max_attempts = 4
    timeout = 30
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            df = yf.download(
                ticker,
                period=period,
                progress=False,
                timeout=timeout,
                auto_adjust=False,
            )
            if df is None or df.empty:
                raise RuntimeError(f"yfinance returned empty dataframe for {ticker}")
            # success
            df.reset_index(inplace=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    c[0] if isinstance(c, tuple) else str(c) for c in df.columns
                ]
            df.rename(columns={"Date": "date", "Adj Close": "Adj_Close"}, inplace=True)
            df["ticker"] = ticker
            path = os.path.join(DATA_DIR, f"{ticker}.parquet")
            df.to_parquet(path)
            return path
        except Exception as e:
            last_err = e
            logger.warning(
                "yfinance download failed for %s (attempt %d/%d): %s",
                ticker,
                attempt,
                max_attempts,
                e,
            )
            if attempt < max_attempts:
                backoff = 2 ** (attempt - 1)
                try:
                    import time

                    time.sleep(backoff)
                except Exception:
                    pass
            else:
                # final failure: raise to let Prefect handle retries / failures
                raise RuntimeError(
                    f"Failed to download data for {ticker} after {max_attempts} attempts: {last_err}"
                )


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
    # remove the ticker column from the persisted feature file to avoid
    # producing trivial `ticker=...` items in downstream association mining
    try:
        df = df.drop(columns=["ticker"])
    except Exception:
        pass
    fpath = os.path.join(FEATURE_DIR, os.path.basename(data_path))
    df.to_parquet(fpath)
    return fpath


@task
def train_regressor(feature_path: str, run_dir: str | None = None):
    df = pd.read_parquet(feature_path)
    y = df["return"].shift(-1)
    feature_cols = [
        c
        for c in df.columns
        if c not in ["date", "ticker", "return"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]
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
    script = os.path.join(
        os.path.dirname(__file__), "..", "ml", "train_classification.py"
    )
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
    min_support: float = 0.05,
    min_confidence: float = 0.5,
    max_rules: int = 200,
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
    feature_cols = [
        c
        for c in df.columns
        if c not in ["date", "ticker"] and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        return {"status": "skipped", "reason": "no numeric features"}

    X_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    # drop constant columns
    nunique = X_df.nunique(dropna=False)
    keep_cols = [c for c in feature_cols if nunique.get(c, 0) > 1]
    removed = [c for c in feature_cols if c not in keep_cols]
    if removed:
        feature_cols = [c for c in feature_cols if c in keep_cols]
        X_df = X_df[feature_cols]

    # log1p Volume to reduce scale/skew
    if "Volume" in X_df.columns:
        try:
            X_df["Volume"] = pd.to_numeric(X_df["Volume"], errors="coerce").fillna(0.0)
            X_df["Volume"] = np.log1p(X_df["Volume"])
        except Exception:
            pass

    X = X_df.dropna()
    if X.empty:
        return {"status": "skipped", "reason": "no data after preprocessing"}

    # scale
    scaler = StandardScaler() if StandardScaler else None
    X_scaled = scaler.fit_transform(X) if scaler is not None else X.to_numpy()

    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    comps = pca.fit_transform(X_scaled)

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
        "mean": scaler.mean_.tolist() if scaler is not None else None,
        "scale": scaler.scale_.tolist() if scaler is not None else None,
        "feature_order": feature_cols,
        "row_index": dates,
        "ticker": ticker,
    }
    # compute and include feature correlation matrix (Pearson) for UI heatmap
    try:
        corr = X_df.corr(method="pearson")
        # represent as nested lists aligned with feature_order
        corr_matrix = (
            corr.reindex(index=feature_cols, columns=feature_cols)
            .fillna(0.0)
            .values.tolist()
        )
        meta["feature_correlation"] = corr_matrix
    except Exception:
        meta["feature_correlation"] = None
    with open(os.path.join(target_dir, f"pca_{ticker}.json"), "w") as f:
        json.dump(meta, f)
    np.save(os.path.join(target_dir, f"pca_transformed_{ticker}.npy"), comps)
    return {"status": "ok", "pca_meta": meta}


@task
def cluster_features(
    feature_path: str, n_clusters: int = 5, run_dir: str | None = None
):
    if KMeans is None:
        return {"status": "skipped", "reason": "sklearn not available"}
    df = pd.read_parquet(feature_path)
    feature_cols = [
        c
        for c in df.columns
        if c not in ["date", "ticker"] and pd.api.types.is_numeric_dtype(df[c])
    ]
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
        "label_counts": {
            int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))
        },
        "ticker": ticker,
    }
    if scaler is not None:
        try:
            meta["scaler_mean"] = scaler.mean_.tolist()
            meta["scaler_scale"] = scaler.scale_.tolist()
        except Exception:
            pass
    # save the original row dates/indices that were used to fit the model so we can align labels later
    try:
        if "date" in df.columns:
            meta["row_index"] = df.loc[X.index, "date"].astype(str).tolist()
        else:
            meta["row_index"] = [str(i) for i in X.index.tolist()]
    except Exception:
        meta["row_index"] = [str(i) for i in X.index.tolist()]
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
    # Need at least 3 dates to infer a frequency; if not available, skip forecasting
    if len(series.index) < 3:
        return {
            "status": "skipped",
            "reason": "insufficient dates to infer frequency for forecasting",
        }
    try:
        freq = pd.infer_freq(series.index)
    except Exception:
        freq = None
    if freq is None:
        # fallback to daily frequency but only if index has reasonable spacing
        freq = "D"
    series = series.asfreq(freq).ffill()
    order = (1, 1, 1)
    logger = logging.getLogger(__name__)
    try:
        # disable automatic re-parameterization enforcement to avoid
        # "Non-stationary starting autoregressive" / "Non-invertible starting MA" warnings
        model = ARIMA(
            series, order=order, enforce_stationarity=False, enforce_invertibility=False
        )

        fitted = None
        # Primary attempt: increase max iterations and capture warnings
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fitted = model.fit(method_kwargs={"maxiter": 500})
                if w:
                    logger.warning(
                        "ARIMA fit produced warnings: %s", [str(x.message) for x in w]
                    )
                logger.debug(
                    "ARIMA mle_retvals (primary): %s",
                    getattr(fitted, "mle_retvals", None),
                )
        except Exception as e1:
            logger.warning("Primary ARIMA fit failed: %s", e1)
            # Fallback: try a different optimizer with more iterations
            try:
                with warnings.catch_warnings(record=True) as w2:
                    warnings.simplefilter("always")
                    fitted = model.fit(method="nm", method_kwargs={"maxiter": 1000})
                    if w2:
                        logger.warning(
                            "ARIMA fallback fit produced warnings: %s",
                            [str(x.message) for x in w2],
                        )
                    logger.debug(
                        "ARIMA mle_retvals (fallback): %s",
                        getattr(fitted, "mle_retvals", None),
                    )
            except Exception as e2:
                logger.exception("ARIMA fit fallback failed: %s", e2)
                # Return a clear skipped result so the pipeline can continue
                return {"status": "skipped", "reason": f"ARIMA fit failed: {e1}; {e2}"}

        # At this point we should have a fitted model
        if fitted is None:
            return {
                "status": "skipped",
                "reason": "ARIMA fit did not produce a fitted model",
            }

        forecast = fitted.forecast(steps=horizon)
        conf_res = fitted.get_forecast(steps=horizon)
        conf = conf_res.conf_int()
    except Exception as e:
        logger.exception("Unexpected error during ARIMA forecasting: %s", e)
        return {"status": "error", "message": str(e)}
    last_date = series.index[-1]
    idx = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=horizon, freq=freq or "D"
    )
    ticker = _ticker_from_path(feature_path)
    target_dir = run_dir or os.path.join(REGISTRY_DIR, ticker)
    os.makedirs(target_dir, exist_ok=True)
    out = {
        "name": "arima-forecast",
        "created_at": datetime.now(UTC).isoformat(),
        "horizon": int(horizon),
        "order": list(order),
        "aic": float(getattr(fitted, "aic", float("nan"))),
        "bic": float(getattr(fitted, "bic", float("nan"))),
        "last_observation": float(series.iloc[-1]),
        "predictions": [float(x) for x in forecast.tolist()],
        "dates": [d.isoformat() for d in idx],
        "confidence_interval": conf.values.tolist(),
        "ticker": ticker,
    }
    # compute diagnostics (log-likelihood, rmse, mae) when possible
    try:
        log_likelihood = float(getattr(fitted, "llf", None))
    except Exception:
        log_likelihood = None
    try:
        resid = np.asarray(getattr(fitted, "resid", []))
        resid = resid[~np.isnan(resid)]
        if resid.size > 0:
            rmse = float(np.sqrt(np.mean(resid**2)))
            mae = float(np.mean(np.abs(resid)))
        else:
            rmse = None
            mae = None
    except Exception:
        rmse = None
        mae = None

    out["log_likelihood"] = log_likelihood
    out["rmse"] = rmse
    out["mae"] = mae
    with open(os.path.join(target_dir, f"forecast_{ticker}.json"), "w") as f:
        json.dump(out, f)
    return {"status": "ok", "forecast": out}


@task
def predict_next(feature_path: str, run_dir: str | None = None):
    """Load the classifier from the run_dir (or registry) and predict the next-day label
    using the latest row of features. Saves a JSON result file in the target dir.
    """
    if xgb is None:
        return {"status": "skipped", "reason": "xgboost not available"}
    df = pd.read_parquet(feature_path).sort_values("date")
    if df.empty:
        return {"status": "skipped", "reason": "no data in features"}
    ticker = _ticker_from_path(feature_path)
    target_dir = run_dir or os.path.join(REGISTRY_DIR, ticker)
    os.makedirs(target_dir, exist_ok=True)
    model_path = os.path.join(target_dir, f"xgb_classifier_{ticker}.ubj")
    meta_path = os.path.join(target_dir, f"xgb_classifier_{ticker}.json")
    # fallback to flat registry if needed
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        model_path = os.path.join(REGISTRY_DIR, f"xgb_classifier_{ticker}.ubj")
        meta_path = os.path.join(REGISTRY_DIR, f"xgb_classifier_{ticker}.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return {"status": "skipped", "reason": "classifier artifact not found"}
    try:
        booster = xgb.Booster()
        booster.load_model(model_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        feats = meta.get("features", [])
        # ensure features exist in df
        missing = [c for c in feats if c not in df.columns]
        if missing:
            return {
                "status": "skipped",
                "reason": f"missing features in feature file: {missing}",
            }
        last_row = df.iloc[-1]
        vals = [float(last_row[c]) for c in feats]
        dmatrix = xgb.DMatrix(pd.DataFrame([vals], columns=feats))
        proba = float(booster.predict(dmatrix)[0])
        label = int(proba >= 0.5)
        out = {
            "ticker": ticker,
            "source_date": str(last_row.get("date", "")),
            "proba_up": proba,
            "label": label,
            "features": feats,
            "values": vals,
        }
        out_path = os.path.join(target_dir, f"predict_next_{ticker}.json")
        with open(out_path, "w") as f:
            json.dump(out, f)
        return {"status": "ok", "prediction_path": out_path, "prediction": out}
    except Exception as e:
        return {"status": "error", "message": str(e)}
