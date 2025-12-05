import os, json
import numpy as np
import pandas as pd
from fastapi import HTTPException
from ..core import XGB_MODEL_PATH, XGB_META_PATH, XGB_CLS_MODEL_PATH, XGB_CLS_META_PATH

try:
    import xgboost as xgb
except Exception:
    xgb = None

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
    meta_feats = [f[0] if isinstance(f, (list, tuple)) else f for f in raw_feats]
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
    clean_cols = {c.strip(): c for c in df.columns}
    assembled = {}
    missing = []
    for bf in booster_feats:
        bf_strip = bf.strip()
        base = bf_strip.split(" ")[0] if " " in bf_strip else bf_strip
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