from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi import Form
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
