from fastapi import FastAPI, Request, UploadFile, File, HTTPException
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
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "ml", "train_baseline.py")

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
