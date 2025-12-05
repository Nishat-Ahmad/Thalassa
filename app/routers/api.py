from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
import os, json, numpy as np, datetime
import pandas as pd
from ..core import (
    MODEL_PATH, XGB_META_PATH, XGB_CLS_META_PATH, PCA_META_PATH, CLUSTERS_META_PATH,
    MODEL_REGISTRY, FORECAST_META_PATH
)
from ..services.models import load_xgb, load_xgb_classifier, align_to_booster_features

try:
    import xgboost as xgb
except Exception:
    xgb = None

router = APIRouter()

class PredictRequest(BaseModel):
    features: list

@router.get("/health")
def health():
    return {"status": "ok", "service": "api", "time": datetime.datetime.utcnow().isoformat()}

@router.get("/model-info")
def model_info():
    meta = {"status": "no-model"}
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "r") as f:
            meta = json.load(f)
    if os.path.exists(XGB_META_PATH):
        with open(XGB_META_PATH, "r") as f:
            meta["xgb"] = json.load(f)
    if os.path.exists(XGB_CLS_META_PATH):
        with open(XGB_CLS_META_PATH, "r") as f:
            meta["xgb_classifier"] = json.load(f)
    if os.path.exists(PCA_META_PATH):
        with open(PCA_META_PATH, "r") as f:
            meta["pca"] = json.load(f)
    if os.path.exists(CLUSTERS_META_PATH):
        with open(CLUSTERS_META_PATH, "r") as f:
            meta["clusters"] = json.load(f)
    if os.path.exists(FORECAST_META_PATH):
        with open(FORECAST_META_PATH, "r") as f:
            meta["forecast"] = json.load(f)
    return meta

@router.get("/expected-features")
def expected_features():
    if not os.path.exists(XGB_META_PATH):
        raise HTTPException(status_code=404, detail="No trained XGB model found")
    with open(XGB_META_PATH, "r") as f:
        meta = json.load(f)
    raw_feats = meta.get("features", [])
    feats = [f[0] if isinstance(f, (list, tuple)) else f for f in raw_feats]
    return {"features": feats}

@router.get("/expected-features-class")
def expected_features_class():
    if not os.path.exists(XGB_CLS_META_PATH):
        raise HTTPException(status_code=404, detail="No trained classifier found")
    with open(XGB_CLS_META_PATH, "r") as f:
        meta = json.load(f)
    feats = [str(f) for f in meta.get("features", [])]
    return {"features": feats}

@router.get("/pca-info")
def pca_info():
    if not os.path.exists(PCA_META_PATH):
        raise HTTPException(status_code=404, detail="PCA metadata not found")
    with open(PCA_META_PATH, "r") as f:
        return json.load(f)

@router.get("/cluster-info")
def cluster_info():
    if not os.path.exists(CLUSTERS_META_PATH):
        raise HTTPException(status_code=404, detail="Cluster metadata not found")
    with open(CLUSTERS_META_PATH, "r") as f:
        meta = json.load(f)
    labels_path = os.path.join(MODEL_REGISTRY, "cluster_labels.npy")
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        meta["latest_label"] = int(labels[-1]) if len(labels) else None
    return meta

@router.post("/predict-cluster")
def predict_cluster(req: PredictRequest):
    if not os.path.exists(CLUSTERS_META_PATH):
        raise HTTPException(status_code=404, detail="Cluster metadata not found")
    with open(CLUSTERS_META_PATH, "r") as f:
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
def predict(req: PredictRequest):
    booster, feat_names = load_xgb()
    if booster is not None and feat_names:
        if len(req.features) != len(feat_names):
            raise HTTPException(status_code=400, detail=f"Expected {len(feat_names)} features, got {len(req.features)}")
        df = pd.DataFrame([req.features], columns=[f.strip() for f in feat_names])
        dmatrix = xgb.DMatrix(df)
        pred = float(booster.predict(dmatrix)[0])
        return {"model": "xgb", "prediction": pred}
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

@router.post("/predict-class")
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

@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    size_kb = round(len(content) / 1024, 2)
    return {"status": "ok", "size_kb": size_kb}

@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    booster, feat_names = load_xgb()
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
def forecast():
    if not os.path.exists(FORECAST_META_PATH):
        raise HTTPException(status_code=404, detail="Forecast not found. Run the pipeline to generate it.")
    with open(FORECAST_META_PATH, "r") as f:
        return json.load(f)

@router.get("/recommend")
def recommend(date: str | None = Query(None), k: int = Query(5, gt=0, le=50), ticker: str = Query("AAPL")):
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
    if date and date in row_index:
        idx = row_index.index(date)
    else:
        idx = len(row_index) - 1
        date = row_index[idx]
    target = comps[idx]
    dists = np.linalg.norm(comps - target, axis=1)
    dists[idx] = np.inf
    nn_idx = np.argsort(dists)[:k]
    feat_path = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "features", f"{ticker}.parquet")
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
def recommend_from_features(req: PredictRequest, k: int = Query(5, gt=0, le=50)):
    if not os.path.exists(PCA_META_PATH):
        raise HTTPException(status_code=404, detail="PCA metadata not found")
    pca_trans_path = os.path.join(MODEL_REGISTRY, "pca_transformed.npy")
    if not os.path.exists(pca_trans_path):
        raise HTTPException(status_code=404, detail="PCA transformed matrix not found. Run pipeline.")
    with open(PCA_META_PATH, "r") as f:
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
