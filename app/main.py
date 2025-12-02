from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi import Form
from fastapi import Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os, json, numpy as np, datetime
import pandas as pd

try:
    import xgboost as xgb
except Exception:
    xgb = None
from subprocess import Popen, PIPE 

APP_TITLE = "Finance Stock Prediction API"
MODEL_REGISTRY = os.path.join(os.path.dirname(__file__), "..", "ml", "registry")
MODEL_PATH = os.path.join(MODEL_REGISTRY, "baseline_model.json")
XGB_META_PATH = os.path.join(MODEL_REGISTRY, "xgb_model.json")
XGB_MODEL_PATH = os.path.join(MODEL_REGISTRY, "xgb_model.ubj")
XGB_CLS_META_PATH = os.path.join(MODEL_REGISTRY, "xgb_classifier.json")
XGB_CLS_MODEL_PATH = os.path.join(MODEL_REGISTRY, "xgb_classifier.ubj")
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "ml", "train_baseline.py")
PCA_META_PATH = os.path.join(MODEL_REGISTRY, "pca.json")
CLUSTERS_META_PATH = os.path.join(MODEL_REGISTRY, "clusters.json")
FORECAST_META_PATH = os.path.join(MODEL_REGISTRY, "forecast.json")
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

class PredictRequest(BaseModel):
    features: list

def load_xgb():
    if xgb is None:
        return None, None
    if not (os.path.exists(XGB_MODEL_PATH) and os.path.exists(XGB_META_PATH)):
        return None, None
    booster = xgb.Booster()
    booster.load_model(XGB_MODEL_PATH)
    with open(XGB_META_PATH, "r") as f:
        meta = json.load(f)
    raw_feats = meta.get("features", [])
    # Normalize meta features to strings
    meta_feats = [f[0] if isinstance(f, (list, tuple)) else f for f in raw_feats]
    # Authoritative feature names from booster (may include ticker suffixes / spaces)
    booster_feats = booster.feature_names or meta_feats
    return booster, booster_feats

def load_xgb_classifier():
    if xgb is None:
        return None, None
    if not (os.path.exists(XGB_CLS_MODEL_PATH) and os.path.exists(XGB_CLS_META_PATH)):
        return None, None
    booster = xgb.Booster()
    booster.load_model(XGB_CLS_MODEL_PATH)
    with open(XGB_CLS_META_PATH, "r") as f:
        meta = json.load(f)
    feats = [str(f) for f in meta.get("features", [])]
    booster_feats = booster.feature_names or feats
    return booster, booster_feats

def align_to_booster_features(df: pd.DataFrame, booster_feats: list[str]) -> pd.DataFrame:
    # Map provided clean columns to booster feature names
    clean_cols = {c.strip(): c for c in df.columns}
    assembled = {}
    missing = []
    for bf in booster_feats:
        bf_strip = bf.strip()
        # Remove common ticker suffix like ' AAPL' if present
        if " " in bf_strip:
            base = bf_strip.split(" ")[0]
        else:
            base = bf_strip
        # Try exact, then stripped, then base
        if bf in df.columns:
            series = df[bf]
        elif bf_strip in clean_cols:
            series = df[clean_cols[bf_strip]]
        elif base in clean_cols:
            series = df[clean_cols[base]]
        else:
            missing.append(bf)
            continue
        assembled[bf] = series
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features required by model: {missing}")
    return pd.DataFrame(assembled)[booster_feats]

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request, "title": "Home", "year": datetime.datetime.now().year}
    )

@app.get("/health")
def health():
    return {"status": "ok", "service": "api", "time": datetime.datetime.utcnow().isoformat()}

@app.get("/model-info")
def model_info():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "r") as f:
            meta = json.load(f)
    else:
        meta = {"status": "no-model"}
    # Overlay XGB info if available
    if os.path.exists(XGB_META_PATH):
        with open(XGB_META_PATH, "r") as f:
            xgb_meta = json.load(f)
        meta["xgb"] = xgb_meta
    if os.path.exists(XGB_CLS_META_PATH):
        with open(XGB_CLS_META_PATH, "r") as f:
            xgb_cls_meta = json.load(f)
        meta["xgb_classifier"] = xgb_cls_meta
    if os.path.exists(PCA_META_PATH):
        with open(PCA_META_PATH, "r") as f:
            meta["pca"] = json.load(f)
    if os.path.exists(CLUSTERS_META_PATH):
        with open(CLUSTERS_META_PATH, "r") as f:
            meta["clusters"] = json.load(f)
    return meta

@app.get("/expected-features")
def expected_features():
    if not os.path.exists(XGB_META_PATH):
        raise HTTPException(status_code=404, detail="No trained XGB model found")
    with open(XGB_META_PATH, "r") as f:
        meta = json.load(f)
    raw_feats = meta.get("features", [])
    feats = [f[0] if isinstance(f, (list, tuple)) else f for f in raw_feats]
    return {"features": feats}

@app.get("/expected-features-class")
def expected_features_class():
    if not os.path.exists(XGB_CLS_META_PATH):
        raise HTTPException(status_code=404, detail="No trained classifier found")
    with open(XGB_CLS_META_PATH, "r") as f:
        meta = json.load(f)
    feats = [str(f) for f in meta.get("features", [])]
    return {"features": feats}

@app.get("/data", response_class=HTMLResponse)
def data_page(request: Request):
    return templates.TemplateResponse("data.html", {"request": request, "title": "Data", "year": datetime.datetime.now().year})

@app.get("/tasks", response_class=HTMLResponse)
def tasks_page(request: Request):
    return templates.TemplateResponse("tasks.html", {"request": request, "title": "Tasks", "year": datetime.datetime.now().year})

@app.get("/roadmap", response_class=HTMLResponse)
def roadmap_page(request: Request):
    return templates.TemplateResponse("roadmap.html", {"request": request, "title": "Roadmap", "year": datetime.datetime.now().year})

@app.get("/contact", response_class=HTMLResponse)
def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request, "title": "Contact", "year": datetime.datetime.now().year})

@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "title": "Upload", "year": datetime.datetime.now().year})

@app.get("/classify", response_class=HTMLResponse)
def classify_page(request: Request):
    feats = []
    if os.path.exists(XGB_CLS_META_PATH):
        with open(XGB_CLS_META_PATH, "r") as f:
            meta = json.load(f)
        feats = [str(f) for f in meta.get("features", [])]
    return templates.TemplateResponse("classify.html", {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": feats})

@app.post("/classify", response_class=HTMLResponse)
def classify_submit(request: Request, values: str = Form(...)):
    booster, feat_names = load_xgb_classifier()
    if booster is None or not feat_names:
        return templates.TemplateResponse("classify.html", {"request": request, "title": "Classify", "year": datetime.datetime.now().year, "features": [], "result": None})
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

@app.get("/pca-info")
def pca_info():
    if not os.path.exists(PCA_META_PATH):
        raise HTTPException(status_code=404, detail="PCA metadata not found")
    with open(PCA_META_PATH, "r") as f:
        return json.load(f)

@app.get("/cluster-info")
def cluster_info():
    if not os.path.exists(CLUSTERS_META_PATH):
        raise HTTPException(status_code=404, detail="Cluster metadata not found")
    with open(CLUSTERS_META_PATH, "r") as f:
        meta = json.load(f)
    # Optionally include most recent cluster label if labels file exists
    labels_path = os.path.join(MODEL_REGISTRY, "cluster_labels.npy")
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        meta["latest_label"] = int(labels[-1]) if len(labels) else None
    return meta

@app.get("/forecast")
def forecast():
    if not os.path.exists(FORECAST_META_PATH):
        raise HTTPException(status_code=404, detail="Forecast not found. Run the pipeline to generate it.")
    with open(FORECAST_META_PATH, "r") as f:
        return json.load(f)

@app.get("/recommend")
def recommend(date: str | None = Query(None), k: int = Query(5, gt=0, le=50), ticker: str = Query("AAPL")):
    # Nearest neighbors in PCA space using persisted transformed matrix.
    if not os.path.exists(PCA_META_PATH):
        raise HTTPException(status_code=404, detail="PCA metadata not found")
    pca_trans_path = os.path.join(MODEL_REGISTRY, "pca_transformed.npy")
    if not os.path.exists(pca_trans_path):
        raise HTTPException(status_code=404, detail="PCA transformed matrix not found. Run pipeline.")
    with open(PCA_META_PATH, "r") as f:
        meta = json.load(f)
    comps = np.load(pca_trans_path)
    row_index = meta.get("row_index", [])
    if not row_index or len(row_index) != len(comps):
        raise HTTPException(status_code=500, detail="Row index missing or misaligned in PCA metadata")
    # Select target row
    if date and date in row_index:
        idx = row_index.index(date)
    else:
        idx = len(row_index) - 1
        date = row_index[idx]
    target = comps[idx]
    # Compute distances to all rows
    dists = np.linalg.norm(comps - target, axis=1)
    # Exclude self
    dists[idx] = np.inf
    nn_idx = np.argsort(dists)[:k]
    # Optionally attach Close values for context
    feat_path = os.path.join(os.path.dirname(__file__), "..", "ml", "features", f"{ticker}.parquet")
    closes = {}
    if os.path.exists(feat_path):
        fdf = pd.read_parquet(feat_path)
        fdf["date"] = pd.to_datetime(fdf["date"]).dt.strftime("%Y-%m-%d")
        closes = dict(zip(fdf["date"], fdf.get("Close", pd.Series([None]*len(fdf)))))
    neighbors = [
        {"date": row_index[i], "distance": float(dists[i]), "close": (None if closes == {} else float(closes.get(row_index[i])) if closes.get(row_index[i]) is not None else None)}
        for i in nn_idx
    ]
    return {"target_date": date, "k": k, "neighbors": neighbors}

@app.post("/recommend")
def recommend_from_features(req: PredictRequest, k: int = Query(5, gt=0, le=50)):
    # Project provided feature vector into PCA space and find nearest historical rows.
    if not os.path.exists(PCA_META_PATH):
        raise HTTPException(status_code=404, detail="PCA metadata not found")
    pca_trans_path = os.path.join(MODEL_REGISTRY, "pca_transformed.npy")
    if not os.path.exists(pca_trans_path):
        raise HTTPException(status_code=404, detail="PCA transformed matrix not found. Run pipeline.")
    with open(PCA_META_PATH, "r") as f:
        meta = json.load(f)
    feat_order = meta.get("feature_order", [])
    mean = np.array(meta.get("mean", []), dtype=float)
    components = np.array(meta.get("components", []), dtype=float)  # shape (n_components, n_features)
    if not feat_order or len(req.features) != len(feat_order):
        raise HTTPException(status_code=400, detail=f"Expected {len(feat_order)} features in PCA feature order")
    if mean.size != len(feat_order) or components.shape[1] != len(feat_order):
        raise HTTPException(status_code=500, detail="PCA metadata incomplete (mean/components)")
    x = np.array(req.features, dtype=float)
    x_centered = x - mean
    # Coordinates in PCA space: (n_features) @ (n_features x n_components)^T => (n_components)
    z = np.dot(x_centered, components.T)
    comps = np.load(pca_trans_path)
    row_index = meta.get("row_index", [])
    if not row_index or len(row_index) != len(comps):
        raise HTTPException(status_code=500, detail="Row index missing or misaligned in PCA metadata")
    dists = np.linalg.norm(comps - z, axis=1)
    nn_idx = np.argsort(dists)[:k]
    neighbors = [
        {"date": row_index[i], "distance": float(dists[i])}
        for i in nn_idx
    ]
    return {"k": k, "neighbors": neighbors}

@app.get("/forecast-page", response_class=HTMLResponse)
def forecast_page(request: Request):
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
        {"request": request, "title": "Forecast", "year": datetime.datetime.now().year, "forecast": data, "error": error}
    )

@app.post("/forecast-page", response_class=HTMLResponse)
def forecast_page_submit(request: Request, horizon: int = Form(7)):
    error = None
    result = None
    # Attempt on-demand ARIMA using existing engineered AAPL features
    if ARIMA is None:
        error = "statsmodels not installed on server; cannot compute on-demand forecast." 
    else:
        feature_path = os.path.join(os.path.dirname(__file__), "..", "ml", "features", "AAPL.parquet")
        if not os.path.exists(feature_path):
            error = "Features file missing (AAPL.parquet). Run pipeline first." 
        else:
            try:
                df = pd.read_parquet(feature_path).sort_values("date")
                if "Close" not in df.columns:
                    error = "Close column missing in features." 
                else:
                    series = df["Close"].astype(float)
                    order = (1,1,1)
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
    # Load persisted forecast again for context
    persisted = None
    if os.path.exists(FORECAST_META_PATH):
        try:
            with open(FORECAST_META_PATH, "r") as f:
                persisted = json.load(f)
        except Exception:
            pass
    return templates.TemplateResponse(
        "forecast.html",
        {"request": request, "title": "Forecast", "year": datetime.datetime.now().year, "forecast": persisted, "computed": result, "error": error}
    )

@app.get("/recommend-page", response_class=HTMLResponse)
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
        },
    )

@app.post("/recommend-page", response_class=HTMLResponse)
def recommend_page_submit(
    request: Request,
    mode: str = Form("date"),
    date: str | None = Form(None),
    k: int = Form(5),
    values: str | None = Form(None),
):
    pca_meta = None
    error = None
    dates = []
    features = []
    neighbors = None
    # Load PCA metadata and transformed matrix
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
            # date mode
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
                # Optional Close values
                feat_path = os.path.join(os.path.dirname(__file__), "..", "ml", "features", "AAPL.parquet")
                closes = {}
                if os.path.exists(feat_path):
                    fdf = pd.read_parquet(feat_path)
                    fdf["date"] = pd.to_datetime(fdf["date"]).dt.strftime("%Y-%m-%d")
                    closes = dict(zip(fdf["date"], fdf.get("Close", pd.Series([None]*len(fdf)))))
                neighbors = [
                    {
                        "date": row_index[i],
                        "distance": float(dists[i]),
                        "close": (None if closes == {} else float(closes.get(row_index[i])) if closes.get(row_index[i]) is not None else None),
                    }
                    for i in nn_idx
                ]
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
        },
    )

@app.post("/predict-cluster")
def predict_cluster(req: PredictRequest):
    # Load cluster metadata and assign cluster for provided feature row using nearest center.
    if not os.path.exists(CLUSTERS_META_PATH):
        raise HTTPException(status_code=404, detail="Cluster metadata not found")
    with open(CLUSTERS_META_PATH, "r") as f:
        meta = json.load(f)
    centers = np.array(meta.get("centers", []), dtype=float)
    feat_order = meta.get("feature_order", [])
    if len(req.features) != len(feat_order):
        raise HTTPException(status_code=400, detail=f"Expected {len(feat_order)} features, got {len(req.features)}")
    x = np.array(req.features, dtype=float)
    # If scaler params are available, standardize input to the same space as centers
    mean = meta.get("scaler_mean")
    scale = meta.get("scaler_scale")
    if isinstance(mean, list) and isinstance(scale, list) and len(mean) == len(x) and len(scale) == len(x):
        mean_arr = np.array(mean, dtype=float)
        scale_arr = np.array(scale, dtype=float)
        # Avoid division by zero
        scale_arr[scale_arr == 0] = 1.0
        x = (x - mean_arr) / scale_arr
    dists = np.linalg.norm(centers - x, axis=1)
    assigned = int(np.argmin(dists))
    return {"cluster": assigned, "distances": dists.tolist()}

@app.get("/pca", response_class=HTMLResponse)
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
    return templates.TemplateResponse(
        "pca.html",
        {
            "request": request,
            "title": "PCA",
            "year": datetime.datetime.now().year,
            "pca": meta,
            "error": error,
        },
    )

@app.get("/cluster", response_class=HTMLResponse)
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
        },
    )

@app.post("/cluster", response_class=HTMLResponse)
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
        },
    )

@app.post("/predict")
def predict(req: PredictRequest):
    # Prefer trained XGB model if available; fallback to baseline dot-product
    booster, feat_names = load_xgb()
    if booster is not None and feat_names:
        # Expect features as list matching feat_names length
        if len(req.features) != len(feat_names):
            raise HTTPException(status_code=400, detail=f"Expected {len(feat_names)} features, got {len(req.features)}")
        df = pd.DataFrame([req.features], columns=[f.strip() for f in feat_names])
        dmatrix = xgb.DMatrix(df)
        pred = float(booster.predict(dmatrix)[0])
        return {"model": "xgb", "prediction": pred}
    # Fallback baseline
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "r") as f:
            meta = json.load(f)
        weights = np.array(meta.get("weights", []), dtype=float)
        if len(weights) != len(req.features):
            weights = np.ones(len(req.features), dtype=float)
    else:
        weights = np.ones(len(req.features), dtype=float)
    x = np.array(req.features, dtype=float)
    y = float(np.dot(x, weights))
    return {"model": "baseline", "prediction": y}

@app.post("/predict-class")
def predict_class(req: PredictRequest):
    booster, feat_names = load_xgb_classifier()
    if booster is None or not feat_names:
        raise HTTPException(status_code=400, detail="Classifier not available. Train it first.")
    if len(req.features) != len(feat_names):
        raise HTTPException(status_code=400, detail=f"Expected {len(feat_names)} features, got {len(req.features)}")
    df = pd.DataFrame([req.features], columns=[f.strip() for f in feat_names])
    dmatrix = xgb.DMatrix(df)
    proba = float(booster.predict(dmatrix)[0])
    label = int(proba >= 0.5)
    return {"model": "xgb_classifier", "proba_up": proba, "label": label}

@app.post("/upload", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    size_kb = round(len(content) / 1024, 2)
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "title": "Upload", "year": datetime.datetime.now().year}
    )

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    booster, feat_names = load_xgb()
    if booster is None or not feat_names:
        raise HTTPException(status_code=400, detail="XGB model not available. Train it first.")
    try:
        df = pd.read_csv(file.file)
    except Exception:
        # Reset and read bytes if needed
        content = await file.read()
        from io import BytesIO
        df = pd.read_csv(BytesIO(content))
    # Align uploaded CSV to booster feature names
    df_aligned = align_to_booster_features(df, feat_names)
    dmatrix = xgb.DMatrix(df_aligned)
    preds = booster.predict(dmatrix)
    return {"count": int(len(preds)), "predictions": preds.tolist()}

@app.post("/train")
def train():
    # Run the training script
    process = Popen(["python", TRAIN_SCRIPT], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        return {"status": "error", "message": stderr.decode()}
    
    return {"status": "success", "message": stdout.decode()}
