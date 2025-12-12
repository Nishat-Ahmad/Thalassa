from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Request, Form
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import os, json, numpy as np, datetime
import sys
from pathlib import Path
import pandas as pd
from ..core import (
    MODEL_REGISTRY,
    templates,
    xgb_paths,
    xgb_classifier_paths,
    pca_paths,
    cluster_paths,
    forecast_path,
    association_path,
)
from ..services.models import load_xgb, load_xgb_classifier, align_to_booster_features

try:
    import xgboost as xgb
except Exception:
    xgb = None

router = APIRouter()
APP_START = datetime.datetime.utcnow()


def _safe_ticker(ticker: str | None) -> str:
    return (ticker or "AAPL").upper()


def _load_pipeline():
    # Lazily import pipeline to avoid package path issues and heavy deps on startup.
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from flows.flow import pipeline  # type: ignore
    return pipeline


def _uptime_seconds() -> float:
    return float((datetime.datetime.utcnow() - APP_START).total_seconds())


def _humanize(seconds: float) -> str:
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hrs:
        parts.append(f"{hrs}h")
    if mins:
        parts.append(f"{mins}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


@router.post("/run-pipeline")
def run_pipeline(
    request: Request,
    ticker_query: str | None = Query(None, alias="ticker"),
    ticker_form: str | None = Form(None, alias="ticker"),
):
    t = _safe_ticker(ticker_form or ticker_query)
    try:
        pipeline = _load_pipeline()
        ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        result = pipeline(t, run_dir=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed for {t}: {e}")

    payload = {"status": "ok", "ticker": t, "result_keys": list(result.keys()) if result else [], "ts": ts}

    accept = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    if "text/html" in accept or content_type.startswith("application/x-www-form-urlencoded"):
        return RedirectResponse(url=f"/tasks?ticker={t}&ran=1&ts={ts}", status_code=303)
    return payload


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _collect_artifacts(ticker: str | None = None):
    items = []

    xgb_model_path, xgb_meta_path = xgb_paths(ticker)
    xgb_cls_model_path, xgb_cls_meta_path = xgb_classifier_paths(ticker)
    pca_meta_path, pca_transformed_path = pca_paths(ticker)
    clusters_meta_path, _ = cluster_paths(ticker)
    forecast_meta_path = forecast_path(ticker)

    def add(name: str, path: str, extractor=None):
        if os.path.exists(path):
            detail = None
            if extractor:
                try:
                    detail = extractor()
                except Exception:
                    detail = None
            items.append({"name": name, "status": "ready", "detail": detail})
        else:
            items.append({"name": name, "status": "missing", "detail": None})

    xgb_meta = _load_json(xgb_meta_path)
    xgb_cls_meta = _load_json(xgb_cls_meta_path)
    pca_meta = _load_json(pca_meta_path)
    cluster_meta = _load_json(clusters_meta_path)
    forecast_meta = _load_json(forecast_meta_path)

    add("XGB Regressor", xgb_model_path, lambda: f"file size {round(os.path.getsize(xgb_model_path)/1024,1)} KB")
    add("XGB Regressor Meta", xgb_meta_path, lambda: f"{len(xgb_meta.get('features', [])) if xgb_meta else 0} features")
    add("Classifier Meta", xgb_cls_meta_path, lambda: f"{len(xgb_cls_meta.get('features', [])) if xgb_cls_meta else 0} features")
    add("PCA Metadata", pca_meta_path, lambda: f"{len(pca_meta.get('feature_order', [])) if pca_meta else 0} feature dims")
    add("Cluster Metadata", clusters_meta_path, lambda: f"{len(cluster_meta.get('centers', [])) if cluster_meta else 0} centers")
    add("Forecast", forecast_meta_path, lambda: f"generated {forecast_meta.get('generated_at', 'unknown') if forecast_meta else 'unknown'}")

    return items

class PredictRequest(BaseModel):
    features: list

@router.get("/health")
def health(request: Request):
    payload = {
        "status": "ok",
        "service": "api",
        "time": datetime.datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": round(_uptime_seconds(), 1),
        "uptime_human": _humanize(_uptime_seconds()),
    }

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return templates.TemplateResponse(
            "health.html",
            {
                "request": request,
                "title": "Health",
                "year": datetime.datetime.now().year,
                **payload,
            },
        )
    return payload

@router.get("/model-info")
def model_info(request: Request, ticker: str = Query("AAPL")):
    t = _safe_ticker(ticker)
    xgb_model_path, xgb_meta_path = xgb_paths(t)
    xgb_cls_model_path, xgb_cls_meta_path = xgb_classifier_paths(t)
    pca_meta_path, _ = pca_paths(t)
    cluster_meta_path, _ = cluster_paths(t)
    forecast_meta_path = forecast_path(t)

    xgb_meta = _load_json(xgb_meta_path)
    xgb_cls_meta = _load_json(xgb_cls_meta_path)
    pca_meta = _load_json(pca_meta_path)
    cluster_meta = _load_json(cluster_meta_path)
    forecast_meta = _load_json(forecast_meta_path)
    artifacts = _collect_artifacts(t)

    legacy_meta: dict = {}
    if xgb_meta:
        legacy_meta["xgb"] = xgb_meta
    if xgb_cls_meta:
        legacy_meta["xgb_classifier"] = xgb_cls_meta
    if pca_meta:
        legacy_meta["pca"] = pca_meta
    if cluster_meta:
        legacy_meta["clusters"] = cluster_meta
    if forecast_meta:
        legacy_meta["forecast"] = forecast_meta

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return templates.TemplateResponse(
            "model_info.html",
            {
                "request": request,
                "title": "Model Info",
                "year": datetime.datetime.now().year,
                "xgb_meta": xgb_meta,
                "xgb_cls_meta": xgb_cls_meta,
                "pca_meta": pca_meta,
                "cluster_meta": cluster_meta,
                "forecast_meta": forecast_meta,
                "artifacts": artifacts,
                "ticker": t,
            },
        )

    return legacy_meta

@router.get("/expected-features")
def expected_features(ticker: str = Query("AAPL")):
    t = _safe_ticker(ticker)
    _, meta_path = xgb_paths(t)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="No trained XGB model found")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    raw_feats = meta.get("features", [])
    feats = [f[0] if isinstance(f, (list, tuple)) else f for f in raw_feats]
    return {"features": feats}

@router.get("/expected-features-class")
def expected_features_class(ticker: str = Query("AAPL")):
    t = _safe_ticker(ticker)
    _, meta_path = xgb_classifier_paths(t)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="No trained classifier found")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feats = [str(f) for f in meta.get("features", [])]
    return {"features": feats}

@router.get("/pca-info")
def pca_info(ticker: str = Query("AAPL")):
    t = _safe_ticker(ticker)
    meta_path, _ = pca_paths(t)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="PCA metadata not found")
    with open(meta_path, "r") as f:
        return json.load(f)

@router.get("/cluster-info")
def cluster_info(ticker: str = Query("AAPL")):
    t = _safe_ticker(ticker)
    meta_path, labels_path = cluster_paths(t)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Cluster metadata not found")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        meta["latest_label"] = int(labels[-1]) if len(labels) else None
    return meta

@router.post("/predict-cluster")
def predict_cluster(req: PredictRequest, ticker: str = Query("AAPL")):
    t = _safe_ticker(ticker)
    meta_path, _ = cluster_paths(t)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Cluster metadata not found")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    centers = np.array(meta.get("centers", []), dtype=float)
    feat_order = meta.get("feature_order", [])
    if len(req.features) != len(feat_order):
        raise HTTPException(status_code=400, detail=f"Expected {len(feat_order)} features, got {len(req.features)}")
    x = np.array(req.features, dtype=float)
    mean = meta.get("scaler_mean")
    scale = meta.get("scaler_scale")
    if isinstance(mean, list) and isinstance(scale, list) and len(mean) == len(x) and len(scale) == len(x):
        mean_arr = np.array(mean, dtype=float)
        scale_arr = np.array(scale, dtype=float)
        scale_arr[scale_arr == 0] = 1.0
        x = (x - mean_arr) / scale_arr
    dists = np.linalg.norm(centers - x, axis=1)
    assigned = int(np.argmin(dists))
    return {"cluster": assigned, "distances": dists.tolist()}

@router.post("/predict")
def predict(req: PredictRequest, ticker: str = Query("AAPL")):
    booster, feat_names = load_xgb(_safe_ticker(ticker))
    if booster is None or not feat_names:
        raise HTTPException(status_code=400, detail="XGB model not available. Train it first.")
    if len(req.features) != len(feat_names):
        raise HTTPException(status_code=400, detail=f"Expected {len(feat_names)} features, got {len(req.features)}")
    df = pd.DataFrame([req.features], columns=[f.strip() for f in feat_names])
    dmatrix = xgb.DMatrix(df)
    pred = float(booster.predict(dmatrix)[0])
    return {"model": "xgb", "prediction": pred}

@router.post("/predict-class")
def predict_class(req: PredictRequest, ticker: str = Query("AAPL")):
    booster, feat_names = load_xgb_classifier(_safe_ticker(ticker))
    if booster is None or not feat_names:
        raise HTTPException(status_code=400, detail="Classifier not available. Train it first.")
    if len(req.features) != len(feat_names):
        raise HTTPException(status_code=400, detail=f"Expected {len(feat_names)} features, got {len(req.features)}")
    df = pd.DataFrame([req.features], columns=[f.strip() for f in feat_names])
    dmatrix = xgb.DMatrix(df)
    proba = float(booster.predict(dmatrix)[0])
    label = int(proba >= 0.5)
    return {"model": "xgb_classifier", "proba_up": proba, "label": label}

@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    size_kb = round(len(content) / 1024, 2)
    return {"status": "ok", "size_kb": size_kb}

@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...), ticker: str = Query("AAPL")):
    booster, feat_names = load_xgb(_safe_ticker(ticker))
    if booster is None or not feat_names:
        raise HTTPException(status_code=400, detail="XGB model not available. Train it first.")
    try:
        df = pd.read_csv(file.file)
    except Exception:
        content = await file.read()
        from io import BytesIO
        df = pd.read_csv(BytesIO(content))
    df_aligned = align_to_booster_features(df, feat_names)
    dmatrix = xgb.DMatrix(df_aligned)
    preds = booster.predict(dmatrix)
    return {"count": int(len(preds)), "predictions": preds.tolist()}

@router.get("/forecast")
def forecast(ticker: str = Query("AAPL")):
    t = _safe_ticker(ticker)
    forecast_meta_path = forecast_path(t)
    if not os.path.exists(forecast_meta_path):
        raise HTTPException(status_code=404, detail="Forecast not found. Run the pipeline to generate it.")
    with open(forecast_meta_path, "r") as f:
        return json.load(f)

@router.get("/recommend")
def recommend(date: str | None = Query(None), k: int = Query(5, gt=0, le=50), ticker: str = Query("AAPL")):
    t = _safe_ticker(ticker)
    pca_meta_path, pca_trans_path = pca_paths(t)
    if not os.path.exists(pca_meta_path):
        raise HTTPException(status_code=404, detail="PCA metadata not found")
    if not os.path.exists(pca_trans_path):
        raise HTTPException(status_code=404, detail="PCA transformed matrix not found. Run pipeline.")
    with open(pca_meta_path, "r") as f:
        meta = json.load(f)
    comps = np.load(pca_trans_path)
    row_index = meta.get("row_index", [])
    if not row_index or len(row_index) != len(comps):
        raise HTTPException(status_code=500, detail="Row index missing or misaligned in PCA metadata")
    if date and date in row_index:
        idx = row_index.index(date)
    else:
        idx = len(row_index) - 1
        date = row_index[idx]
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
    neighbors = [
        {"date": row_index[i], "distance": float(dists[i]), "close": (None if closes == {} else float(closes.get(row_index[i])) if closes.get(row_index[i]) is not None else None)}
        for i in nn_idx
    ]
    return {"target_date": date, "k": k, "neighbors": neighbors}

@router.post("/recommend")
def recommend_from_features(req: PredictRequest, k: int = Query(5, gt=0, le=50), ticker: str = Query("AAPL")):
    t = _safe_ticker(ticker)
    pca_meta_path, pca_trans_path = pca_paths(t)
    if not os.path.exists(pca_meta_path):
        raise HTTPException(status_code=404, detail="PCA metadata not found")
    if not os.path.exists(pca_trans_path):
        raise HTTPException(status_code=404, detail="PCA transformed matrix not found. Run pipeline.")
    with open(pca_meta_path, "r") as f:
        meta = json.load(f)
    feat_order = meta.get("feature_order", [])
    mean = np.array(meta.get("mean", []), dtype=float)
    components = np.array(meta.get("components", []), dtype=float)
    if not feat_order or len(req.features) != len(feat_order):
        raise HTTPException(status_code=400, detail=f"Expected {len(feat_order)} features in PCA feature order")
    if mean.size != len(feat_order) or components.shape[1] != len(feat_order):
        raise HTTPException(status_code=500, detail="PCA metadata incomplete (mean/components)")
    x = np.array(req.features, dtype=float)
    z = np.dot(x - mean, components.T)
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

@router.get("/association-info")
def association_info(ticker: str = Query("AAPL")):
    path = association_path(_safe_ticker(ticker))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Association rules not found. Run association flow.")
    with open(path, "r") as f:
        return json.load(f)
