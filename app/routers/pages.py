from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
import os, json, math, numpy as np, datetime
import pandas as pd
import yfinance as yf
from ..core import (
    templates,
    MODEL_REGISTRY,
    xgb_classifier_paths,
    pca_paths,
    cluster_paths,
    forecast_path,
    association_path,
)

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None
try:
    import xgboost as xgb
except Exception:
    xgb = None

router = APIRouter()


def _safe_ticker(ticker: str | None) -> str:
    return (ticker or "AAPL").upper()

@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request, "title": "Home", "year": datetime.datetime.now().year}
    )

@router.get("/data", response_class=HTMLResponse)
def data_page(request: Request):
    return templates.TemplateResponse("data.html", {"request": request, "title": "Data", "year": datetime.datetime.now().year})

@router.get("/search", response_class=HTMLResponse)
def search_page(request: Request, ticker: str | None = None, period: str | None = None):
    ticker_info = None
    recent = None
    error = None
    suggestions = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "NFLX"]
    period_map = {
        "1w": "7d",
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y",
    }
    chosen_period = period if period in period_map else "1mo"
    if ticker:
        t = ticker.strip().upper()
        try:
            tk = yf.Ticker(t)
            info = tk.fast_info if hasattr(tk, "fast_info") else None
            details = {}
            try:
                details = tk.get_info() or {}
            except Exception:
                details = {}

            def r3(val):
                try:
                    f = float(val)
                    if math.isfinite(f):
                        return round(f, 3)
                except Exception:
                    return None
                return None
            ticker_info = {
                "symbol": t,
                "name": details.get("longName") or details.get("shortName"),
                "currency": getattr(info, "currency", None) if info else details.get("currency"),
                "last_price": r3(getattr(info, "last_price", None) if info else details.get("currentPrice")),
                "previous_close": r3(getattr(info, "previous_close", None) if info else details.get("previousClose")),
                "year_high": r3(getattr(info, "year_high", None) if info else details.get("fiftyTwoWeekHigh")),
                "year_low": r3(getattr(info, "year_low", None) if info else details.get("fiftyTwoWeekLow")),
            }
            df = tk.history(period=period_map[chosen_period])
            if not df.empty:
                df = df.reset_index()
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
                for col in ["Open", "High", "Low", "Close"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").round(3)
                recent = df.to_dict(orient="records")
        except Exception as e:
            error = f"Lookup failed for {t}: {e}"
    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "title": "Stock Search",
            "year": datetime.datetime.now().year,
            "ticker": ticker or "",
            "period": chosen_period,
            "period_options": list(period_map.keys()),
            "ticker_info": ticker_info,
            "recent": recent,
            "error": error,
            "suggestions": suggestions,
        },
    )

@router.get("/tasks", response_class=HTMLResponse)
def tasks_page(request: Request, ticker: str | None = None):
    t = _safe_ticker(ticker)
    return templates.TemplateResponse(
        "tasks.html",
        {"request": request, "title": "Tasks", "year": datetime.datetime.now().year, "ticker": t},
    )

# Roadmap page removed per request

@router.get("/contact", response_class=HTMLResponse)
def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request, "title": "Support", "year": datetime.datetime.now().year})

@router.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "title": "Upload", "year": datetime.datetime.now().year})

@router.get("/classify", response_class=HTMLResponse)
def classify_page(request: Request, ticker: str | None = None):
    t = _safe_ticker(ticker)
    _, cls_meta_path = xgb_classifier_paths(t)
    feats = []
    if os.path.exists(cls_meta_path):
        with open(cls_meta_path, "r") as f:
            meta = json.load(f)
        feats = [str(f) for f in meta.get("features", [])]
    return templates.TemplateResponse(
        "classify.html",
        {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": feats, "ticker": t},
    )

@router.post("/classify", response_class=HTMLResponse)
def classify_submit(request: Request, values: str = Form(...), ticker: str | None = None):
    t = _safe_ticker(ticker)
    cls_model_path, cls_meta_path = xgb_classifier_paths(t)
    if xgb is None or not (os.path.exists(cls_meta_path) and os.path.exists(cls_model_path)):
        return templates.TemplateResponse(
            "classify.html",
            {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": [], "result": None, "ticker": t},
        )
    booster = xgb.Booster()
    booster.load_model(cls_model_path)
    with open(cls_meta_path, "r") as f:
        meta = json.load(f)
    feat_names = [str(f) for f in meta.get("features", [])]
    try:
        nums = [float(x.strip()) for x in values.split(",") if x.strip() != ""]
    except Exception:
        return templates.TemplateResponse(
            "classify.html",
            {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": feat_names, "result": None, "ticker": t},
        )
    if len(nums) != len(feat_names):
        return templates.TemplateResponse(
            "classify.html",
            {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": feat_names, "result": None, "ticker": t},
        )
    df = pd.DataFrame([nums], columns=[f.strip() for f in feat_names])
    dmatrix = xgb.DMatrix(df)
    proba = float(booster.predict(dmatrix)[0])
    result = {"proba_up": proba, "label": int(proba >= 0.5)}
    return templates.TemplateResponse(
        "classify.html",
        {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": feat_names, "result": result, "ticker": t},
    )

@router.get("/pca", response_class=HTMLResponse)
def pca_page(request: Request, ticker: str | None = None):
    t = _safe_ticker(ticker)
    meta_path, _ = pca_paths(t)
    meta = None
    error = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as e:
            error = f"Failed to load PCA metadata: {e}"
    else:
        error = "PCA metadata not found. Run the pipeline to generate it."
    return templates.TemplateResponse(
        "pca.html",
        {"request": request, "title": "PCA", "year": datetime.datetime.now().year, "pca": meta, "error": error, "ticker": t},
    )

@router.get("/cluster", response_class=HTMLResponse)
def cluster_page(request: Request, ticker: str | None = None):
    t = _safe_ticker(ticker)
    meta_path, _ = cluster_paths(t)
    meta = None
    error = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as e:
            error = f"Failed to load cluster metadata: {e}"
    else:
        error = "Cluster metadata not found. Run the pipeline to generate it."
    features = meta.get("feature_order", []) if isinstance(meta, dict) else []
    return templates.TemplateResponse(
        "cluster.html",
        {
            "request": request,
            "title": "Clustering",
            "year": datetime.datetime.now().year,
            "clusters": meta,
            "features": features,
            "result": None,
            "error": error,
            "ticker": t,
        },
    )

@router.post("/cluster", response_class=HTMLResponse)
def cluster_submit(request: Request, values: str = Form(...), ticker: str | None = None):
    t = _safe_ticker(ticker)
    meta_path, _ = cluster_paths(t)
    meta = None
    error = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        error = "Cluster metadata not found. Run the pipeline to generate it."
    features = meta.get("feature_order", []) if isinstance(meta, dict) else []
    result = None
    if not error and meta:
        try:
            nums = [float(x.strip()) for x in values.split(",") if x.strip() != ""]
        except Exception:
            nums = []
        if len(nums) == len(features):
            centers = np.array(meta.get("centers", []), dtype=float)
            x = np.array(nums, dtype=float)
            mean = meta.get("scaler_mean")
            scale = meta.get("scaler_scale")
            if isinstance(mean, list) and isinstance(scale, list) and len(mean) == len(x) and len(scale) == len(x):
                mean_arr = np.array(mean, dtype=float)
                scale_arr = np.array(scale, dtype=float)
                scale_arr[scale_arr == 0] = 1.0
                x = (x - mean_arr) / scale_arr
            dists = np.linalg.norm(centers - x, axis=1)
            assigned = int(np.argmin(dists))
            result = {"cluster": assigned, "distances": dists.tolist()}
        else:
            error = f"Expected {len(features)} features, got {len(nums)}"
    return templates.TemplateResponse(
        "cluster.html",
        {
            "request": request,
            "title": "Clustering",
            "year": datetime.datetime.now().year,
            "clusters": meta,
            "features": features,
            "result": result,
            "error": error,
            "ticker": t,
        },
    )

@router.get("/forecast-page", response_class=HTMLResponse)
def forecast_page(request: Request, ticker: str | None = None):
    t = _safe_ticker(ticker)
    FORECAST_META_PATH = forecast_path(t)
    data = None
    error = None
    if os.path.exists(FORECAST_META_PATH):
        try:
            with open(FORECAST_META_PATH, "r") as f:
                data = json.load(f)
        except Exception as e:
            error = f"Failed to load forecast: {e}"
    else:
        error = "No persisted forecast found. Run pipeline or submit a horizon to compute one."
    return templates.TemplateResponse(
        "forecast.html",
        {"request": request, "title": "Forecast", "year": datetime.datetime.now().year, "forecast": data, "error": error, "ticker": t},
    )

@router.post("/forecast-page", response_class=HTMLResponse)
def forecast_page_submit(request: Request, horizon: int = Form(7), ticker: str | None = None):
    t = _safe_ticker(ticker)
    error = None
    result = None
    if ARIMA is None:
        error = "statsmodels not installed on server; cannot compute on-demand forecast."
    else:
        feature_path = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet")
        if not os.path.exists(feature_path):
            error = f"Features file missing ({t}.parquet). Run pipeline first."
        else:
            try:
                df = pd.read_parquet(feature_path).sort_values("date")
                if "Close" not in df.columns:
                    error = "Close column missing in features."
                else:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.dropna(subset=["date"])
                    series = df.set_index("date")["Close"].astype(float).sort_index()
                    freq = pd.infer_freq(series.index)
                    if freq is None:
                        freq = "D"
                    series = series.asfreq(freq).ffill()
                    order = (1, 1, 1)
                    model = ARIMA(series, order=order)
                    fitted = model.fit()
                    fc_vals = fitted.forecast(steps=horizon)
                    conf_res = fitted.get_forecast(steps=horizon)
                    conf = conf_res.conf_int().values.tolist()
                    last_date = series.index[-1]
                    idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq=freq or "D")
                    result = {
                        "horizon": int(horizon),
                        "order": list(order),
                        "aic": float(getattr(fitted, 'aic', float('nan'))),
                        "bic": float(getattr(fitted, 'bic', float('nan'))),
                        "last_observation": float(series.iloc[-1]),
                        "dates": [d.isoformat() for d in idx],
                        "predictions": [float(x) for x in fc_vals.tolist()],
                        "confidence_interval": conf,
                    }
            except Exception as e:
                error = f"Forecast error: {e}"
    persisted = None
    FORECAST_META_PATH = forecast_path(t)
    if os.path.exists(FORECAST_META_PATH):
        try:
            with open(FORECAST_META_PATH, "r") as f:
                persisted = json.load(f)
        except Exception:
            pass
    return templates.TemplateResponse(
        "forecast.html",
        {
            "request": request,
            "title": "Forecast",
            "year": datetime.datetime.now().year,
            "forecast": persisted,
            "computed": result,
            "error": error,
            "ticker": t,
        },
    )

@router.get("/recommend-page", response_class=HTMLResponse)
def recommend_page(request: Request, ticker: str | None = None):
    t = _safe_ticker(ticker)
    pca_meta_path, _ = pca_paths(t)
    pca_meta = None
    dates = []
    features = []
    error = None
    if os.path.exists(pca_meta_path):
        try:
            with open(pca_meta_path, "r") as f:
                pca_meta = json.load(f)
            dates = pca_meta.get("row_index", [])[-30:]
            features = pca_meta.get("feature_order", [])
        except Exception as e:
            error = f"Failed to load PCA metadata: {e}"
    else:
        error = "PCA artifacts not found. Run the pipeline to generate them."
    return templates.TemplateResponse(
        "recommend.html",
        {
            "request": request,
            "title": "Recommend",
            "year": datetime.datetime.now().year,
            "dates": dates,
            "features": features,
            "neighbors": None,
            "error": error,
            "ticker": t,
        },
    )

@router.post("/recommend-page", response_class=HTMLResponse)
def recommend_page_submit(
    request: Request,
    mode: str = Form("date"),
    date: str | None = Form(None),
    k: int = Form(5),
    values: str | None = Form(None),
    ticker: str | None = None,
):
    t = _safe_ticker(ticker)
    pca_meta_path, pca_trans_path = pca_paths(t)
    pca_meta = None
    error = None
    dates = []
    features = []
    neighbors = None
    if not os.path.exists(pca_meta_path):
        error = "PCA metadata not found. Run pipeline first."
    else:
        with open(pca_meta_path, "r") as f:
            pca_meta = json.load(f)
        features = pca_meta.get("feature_order", [])
        all_dates = pca_meta.get("row_index", [])
        dates = all_dates[-30:]
        trans_path = pca_trans_path
        if not os.path.exists(trans_path):
            error = "PCA transformed matrix missing. Run pipeline."
    if error is None and pca_meta is not None:
        comps = np.load(pca_trans_path)
        row_index = pca_meta.get("row_index", [])
        if mode == "vector" and values:
            try:
                nums = [float(x.strip()) for x in values.split(",") if x.strip() != ""]
            except Exception:
                nums = []
            if len(nums) != len(features):
                error = f"Expected {len(features)} features, got {len(nums)}"
            else:
                mean = np.array(pca_meta.get("mean", []), dtype=float)
                components = np.array(pca_meta.get("components", []), dtype=float)
                if mean.size != len(features) or components.shape[1] != len(features):
                    error = "PCA metadata incomplete (mean/components)."
                else:
                    x = np.array(nums, dtype=float)
                    z = np.dot(x - mean, components.T)
                    dists = np.linalg.norm(comps - z, axis=1)
                    nn_idx = np.argsort(dists)[:k]
                    neighbors = [{"date": row_index[i], "distance": float(dists[i])} for i in nn_idx]
        else:
            if not row_index:
                error = "Row index missing in PCA metadata."
            else:
                if (not date) or (date not in row_index):
                    idx = len(row_index) - 1
                    date = row_index[idx]
                else:
                    idx = row_index.index(date)
                target = comps[idx]
                dists = np.linalg.norm(comps - target, axis=1)
                dists[idx] = np.inf
                nn_idx = np.argsort(dists)[:k]
                feat_path = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet")
                closes = {}
                if os.path.exists(feat_path):
                    fdf = pd.read_parquet(feat_path)
                    fdf["date"] = pd.to_datetime(fdf["date"]).dt.strftime("%Y-%m-%d")
                    closes = dict(zip(fdf["date"], fdf.get("Close", pd.Series([None]*len(fdf)))))
                neighbors = [{"date": row_index[i], "distance": float(dists[i]), "close": (None if closes == {} else float(closes.get(row_index[i])) if closes.get(row_index[i]) is not None else None)} for i in nn_idx]
    return templates.TemplateResponse(
        "recommend.html",
        {
            "request": request,
            "title": "Recommend",
            "year": datetime.datetime.now().year,
            "dates": dates,
            "features": features,
            "neighbors": neighbors,
            "error": error,
            "selected_date": date,
            "mode": mode,
            "ticker": t,
        },
    )

@router.get("/association-page", response_class=HTMLResponse)
def association_page(request: Request, ticker: str | None = None):
    t = _safe_ticker(ticker)
    path = association_path(t)
    data = None
    error = None
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            error = f"Failed to load association rules: {e}"
    else:
        error = "No association rules found. Run the association flow to generate them."
    return templates.TemplateResponse(
        "association.html",
        {"request": request, "title": "Association", "year": datetime.datetime.now().year, "assoc": data, "error": error, "ticker": t},
    )
