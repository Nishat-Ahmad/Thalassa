import json
import os

import pandas as pd
from fastapi import HTTPException
from ..core import xgb_paths, xgb_classifier_paths

try:
    import xgboost as xgb
except Exception:
    xgb = None

def load_xgb(ticker: str | None = None):
    if xgb is None:
        return None, None
    model_path, meta_path = xgb_paths(ticker)
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    raw_feats = meta.get("features", [])
    meta_feats = [f[0] if isinstance(f, (list, tuple)) else f for f in raw_feats]
    booster_feats = booster.feature_names or meta_feats
    return booster, booster_feats


def load_xgb_classifier(ticker: str | None = None):
    if xgb is None:
        return None, None
    model_path, meta_path = xgb_classifier_paths(ticker)
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(meta_path, "r") as f:
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