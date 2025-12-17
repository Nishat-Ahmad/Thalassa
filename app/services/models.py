import json
import os
import re
from collections import deque

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pandas.tseries.offsets import BDay
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


def align_to_booster_features(
    df: pd.DataFrame, booster_feats: list[str]
) -> pd.DataFrame:
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
        raise HTTPException(
            status_code=400, detail=f"Missing features required by model: {missing}"
        )
    return pd.DataFrame(assembled)[booster_feats]


def forecast_regressor_next_days(
    df: pd.DataFrame,
    booster,
    feat_names: list[str],
    days: int = 7,
) -> tuple[list[str | None], list[float]]:
    """Approximate multi-step (next-N business days) forecasting for the regressor.

    Uses the latest available feature row and iteratively updates a subset of features
    (lag_close_*, lag_return_*, sma_*, ema_*, vol_10) using predicted closes/returns.

    Returns (predicted_dates, predictions) where predictions are fractional returns.
    """

    if booster is None or not feat_names:
        raise HTTPException(status_code=400, detail="Model not available")

    try:
        days = int(days or 7)
    except Exception:
        days = 7
    if days < 1:
        days = 1
    if days > 90:
        days = 90

    if "Close" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Feature file missing 'Close' column; cannot forecast multiple days",
        )

    # determine last date for labeling
    last_date = None
    if "date" in df.columns:
        try:
            last_date = pd.to_datetime(df["date"].iloc[-1])
        except Exception:
            last_date = None

    # align to model features and take latest feature row
    df_aligned_all = align_to_booster_features(df, feat_names)
    try:
        last_feat = df_aligned_all.tail(1).iloc[0].to_dict()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to select latest features")

    # collect historical windows for updating statistics
    closes_hist = list(pd.to_numeric(df["Close"].dropna()).tail(200))
    returns_hist = list(pd.to_numeric(df.get("return", pd.Series([])).dropna()).tail(200))

    closes = deque(closes_hist[::-1])
    returns = deque(returns_hist[::-1])

    # infer max window/lag needed
    max_needed = 1
    sma_windows: dict[str, int] = {}
    ema_windows: dict[str, int] = {}
    for fn in feat_names:
        m = re.match(r"lag_close_(\d+)$", fn)
        if m:
            max_needed = max(max_needed, int(m.group(1)))
        m = re.match(r"lag_return_(\d+)$", fn)
        if m:
            max_needed = max(max_needed, int(m.group(1)))
        m = re.match(r"sma_(\d+)$", fn)
        if m:
            w = int(m.group(1))
            sma_windows[fn] = w
            max_needed = max(max_needed, w)
        m = re.match(r"ema_(\d+)$", fn)
        if m:
            w = int(m.group(1))
            ema_windows[fn] = w
            max_needed = max(max_needed, w)

    while len(closes) < max_needed:
        closes.append(0.0)
    while len(returns) < max_needed:
        returns.append(0.0)

    current_close = float(closes[0])
    predicted_dates: list[str | None] = []
    preds: list[float] = []

    for step in range(1, days + 1):
        fv: dict[str, float] = {}
        for fn in feat_names:
            m = re.match(r"lag_close_(\d+)$", fn)
            if m:
                k = int(m.group(1))
                fv[fn] = float(closes[k - 1]) if k - 1 < len(closes) else float(closes[-1])
                continue
            m = re.match(r"lag_return_(\d+)$", fn)
            if m:
                k = int(m.group(1))
                fv[fn] = float(returns[k - 1]) if k - 1 < len(returns) else float(returns[-1])
                continue
            if fn in sma_windows:
                w = sma_windows[fn]
                old = float(last_feat.get(fn, 0.0) or 0.0)
                fv[fn] = (old * (w - 1) + current_close) / float(w)
                continue
            if fn in ema_windows:
                w = ema_windows[fn]
                old = float(last_feat.get(fn, 0.0) or 0.0)
                alpha = 2.0 / (w + 1)
                fv[fn] = alpha * current_close + (1 - alpha) * old
                continue
            if fn == "vol_10":
                vals = list(returns)[:10]
                try:
                    fv[fn] = float(np.std(vals)) if vals else float(last_feat.get(fn, 0.0) or 0.0)
                except Exception:
                    fv[fn] = float(last_feat.get(fn, 0.0) or 0.0)
                continue

            fv[fn] = float(last_feat.get(fn, 0.0) or 0.0)

        # predict next-day return (fractional)
        row = pd.DataFrame([fv], columns=[f for f in feat_names])
        dmat = xgb.DMatrix(row)
        pred = float(booster.predict(dmat)[0])
        preds.append(pred)

        # update deques based on predicted close
        predicted_close = current_close * (1.0 + pred)
        closes.appendleft(predicted_close)
        returns.appendleft(pred)
        while len(closes) > max_needed:
            closes.pop()
        while len(returns) > max_needed:
            returns.pop()

        last_feat.update(fv)
        current_close = float(predicted_close)

        if last_date is not None:
            try:
                predicted_dates.append((last_date + BDay(step)).strftime("%Y-%m-%d"))
            except Exception:
                predicted_dates.append(None)
        else:
            predicted_dates.append(None)

    return predicted_dates, preds
