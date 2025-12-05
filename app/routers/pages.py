from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
import os, json, numpy as np, datetime
import pandas as pd
from ..core import templates, PCA_META_PATH, CLUSTERS_META_PATH, MODEL_REGISTRY

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None
try:
    import xgboost as xgb
except Exception:
    xgb = None

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request, "title": "Home", "year": datetime.datetime.now().year}
    )

@router.get("/data", response_class=HTMLResponse)
def data_page(request: Request):
    return templates.TemplateResponse("data.html", {"request": request, "title": "Data", "year": datetime.datetime.now().year})

@router.get("/tasks", response_class=HTMLResponse)
def tasks_page(request: Request):
    return templates.TemplateResponse("tasks.html", {"request": request, "title": "Tasks", "year": datetime.datetime.now().year})

@router.get("/roadmap", response_class=HTMLResponse)
def roadmap_page(request: Request):
    return templates.TemplateResponse("roadmap.html", {"request": request, "title": "Roadmap", "year": datetime.datetime.now().year})

@router.get("/contact", response_class=HTMLResponse)
def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request, "title": "Contact", "year": datetime.datetime.now().year})

@router.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "title": "Upload", "year": datetime.datetime.now().year})

@router.get("/classify", response_class=HTMLResponse)
def classify_page(request: Request):
    XGB_CLS_META_PATH = os.path.join(MODEL_REGISTRY, "xgb_classifier.json")
    feats = []
    if os.path.exists(XGB_CLS_META_PATH):
        with open(XGB_CLS_META_PATH, "r") as f:
            meta = json.load(f)
        feats = [str(f) for f in meta.get("features", [])]
    return templates.TemplateResponse("classify.html", {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": feats})

@router.post("/classify", response_class=HTMLResponse)
def classify_submit(request: Request, values: str = Form(...)):
    XGB_CLS_META_PATH = os.path.join(MODEL_REGISTRY, "xgb_classifier.json")
    XGB_CLS_MODEL_PATH = os.path.join(MODEL_REGISTRY, "xgb_classifier.ubj")
    if xgb is None or not (os.path.exists(XGB_CLS_META_PATH) and os.path.exists(XGB_CLS_MODEL_PATH)):
        return templates.TemplateResponse("classify.html", {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": [], "result": None})
    booster = xgb.Booster()
    booster.load_model(XGB_CLS_MODEL_PATH)
    with open(XGB_CLS_META_PATH, "r") as f:
        meta = json.load(f)
    feat_names = [str(f) for f in meta.get("features", [])]
    try:
        nums = [float(x.strip()) for x in values.split(",") if x.strip() != ""]
    except Exception:
        return templates.TemplateResponse("classify.html", {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": feat_names, "result": None})
    if len(nums) != len(feat_names):
        return templates.TemplateResponse("classify.html", {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": feat_names, "result": None})
    df = pd.DataFrame([nums], columns=[f.strip() for f in feat_names])
    dmatrix = xgb.DMatrix(df)
    proba = float(booster.predict(dmatrix)[0])
    result = {"proba_up": proba, "label": int(proba >= 0.5)}
    return templates.TemplateResponse("classify.html", {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": feat_names, "result": result})

@router.get("/pca", response_class=HTMLResponse)
def pca_page(request: Request):
    meta = None
    error = None
    if os.path.exists(PCA_META_PATH):
        try:
            with open(PCA_META_PATH, "r") as f:
                meta = json.load(f)
        except Exception as e:
            error = f"Failed to load PCA metadata: {e}"
    else:
        error = "PCA metadata not found. Run the pipeline to generate it."
    return templates.TemplateResponse("pca.html", {"request": request, "title": "PCA", "year": datetime.datetime.now().year, "pca": meta, "error": error})

@router.get("/cluster", response_class=HTMLResponse)
def cluster_page(request: Request):
    meta = None
    error = None
    if os.path.exists(CLUSTERS_META_PATH):
        try:
            with open(CLUSTERS_META_PATH, "r") as f:
                meta = json.load(f)
        except Exception as e:
            error = f"Failed to load cluster metadata: {e}"
    else:
        error = "Cluster metadata not found. Run the pipeline to generate it."
    features = meta.get("feature_order", []) if isinstance(meta, dict) else []
    return templates.TemplateResponse("cluster.html", {"request": request, "title": "Clustering", "year": datetime.datetime.now().year, "clusters": meta, "features": features, "result": None, "error": error})

@router.post("/cluster", response_class=HTMLResponse)
def cluster_submit(request: Request, values: str = Form(...)):
    meta = None
    error = None
    if os.path.exists(CLUSTERS_META_PATH):
        with open(CLUSTERS_META_PATH, "r") as f:
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
    return templates.TemplateResponse("cluster.html", {"request": request, "title": "Clustering", "year": datetime.datetime.now().year, "clusters": meta, "features": features, "result": result, "error": error})

@router.get("/forecast-page", response_class=HTMLResponse)
def forecast_page(request: Request):
    FORECAST_META_PATH = os.path.join(MODEL_REGISTRY, "forecast.json")
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
    return templates.TemplateResponse("forecast.html", {"request": request, "title": "Forecast", "year": datetime.datetime.now().year, "forecast": data, "error": error})

@router.post("/forecast-page", response_class=HTMLResponse)
def forecast_page_submit(request: Request, horizon: int = Form(7)):
    error = None
    result = None
    if ARIMA is None:
        error = "statsmodels not installed on server; cannot compute on-demand forecast."
    else:
        feature_path = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "features", "AAPL.parquet")
        if not os.path.exists(feature_path):
            error = "Features file missing (AAPL.parquet). Run pipeline first."
        else:
            try:
                df = pd.read_parquet(feature_path).sort_values("date")
                if "Close" not in df.columns:
                    error = "Close column missing in features."
                else:
                    series = df["Close"].astype(float)
                    order = (1, 1, 1)
                    model = ARIMA(series, order=order)
                    fitted = model.fit()
                    fc_vals = fitted.forecast(steps=horizon)
                    conf_res = fitted.get_forecast(steps=horizon)
                    conf = conf_res.conf_int().values.tolist()
                    last_date = pd.to_datetime(df["date"].iloc[-1])
                    idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
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
    FORECAST_META_PATH = os.path.join(MODEL_REGISTRY, "forecast.json")
    if os.path.exists(FORECAST_META_PATH):
        try:
            with open(FORECAST_META_PATH, "r") as f:
                persisted = json.load(f)
        except Exception:
            pass
    return templates.TemplateResponse("forecast.html", {"request": request, "title": "Forecast", "year": datetime.datetime.now().year, "forecast": persisted, "computed": result, "error": error})

@router.get("/recommend-page", response_class=HTMLResponse)
def recommend_page(request: Request):
    pca_meta = None
    dates = []
    features = []
    error = None
    if os.path.exists(PCA_META_PATH):
        try:
            with open(PCA_META_PATH, "r") as f:
                pca_meta = json.load(f)
            dates = pca_meta.get("row_index", [])[-30:]
            features = pca_meta.get("feature_order", [])
        except Exception as e:
            error = f"Failed to load PCA metadata: {e}"
    else:
        error = "PCA artifacts not found. Run the pipeline to generate them."
    return templates.TemplateResponse("recommend.html", {"request": request, "title": "Recommend", "year": datetime.datetime.now().year, "dates": dates, "features": features, "neighbors": None, "error": error})

@router.post("/recommend-page", response_class=HTMLResponse)
def recommend_page_submit(request: Request, mode: str = Form("date"), date: str | None = Form(None), k: int = Form(5), values: str | None = Form(None)):
    pca_meta = None
    error = None
    dates = []
    features = []
    neighbors = None
    if not os.path.exists(PCA_META_PATH):
        error = "PCA metadata not found. Run pipeline first."
    else:
        with open(PCA_META_PATH, "r") as f:
            pca_meta = json.load(f)
        features = pca_meta.get("feature_order", [])
        all_dates = pca_meta.get("row_index", [])
        dates = all_dates[-30:]
        trans_path = os.path.join(MODEL_REGISTRY, "pca_transformed.npy")
        if not os.path.exists(trans_path):
            error = "PCA transformed matrix missing. Run pipeline."
    if error is None and pca_meta is not None:
        comps = np.load(os.path.join(MODEL_REGISTRY, "pca_transformed.npy"))
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
                feat_path = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "features", "AAPL.parquet")
                closes = {}
                if os.path.exists(feat_path):
                    fdf = pd.read_parquet(feat_path)
                    fdf["date"] = pd.to_datetime(fdf["date"]).dt.strftime("%Y-%m-%d")
                    closes = dict(zip(fdf["date"], fdf.get("Close", pd.Series([None]*len(fdf)))))
                neighbors = [{"date": row_index[i], "distance": float(dists[i]), "close": (None if closes == {} else float(closes.get(row_index[i])) if closes.get(row_index[i]) is not None else None)} for i in nn_idx]
    return templates.TemplateResponse("recommend.html", {"request": request, "title": "Recommend", "year": datetime.datetime.now().year, "dates": dates, "features": features, "neighbors": neighbors, "error": error, "selected_date": date, "mode": mode})
