from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
import datetime
import json
import math
import os
import re

import numpy as np
import pandas as pd
import yfinance as yf
from ..core import (
    templates,
    MODEL_REGISTRY,
    xgb_classifier_paths,
    pca_paths,
    xgb_paths,
    cluster_paths,
    forecast_path,
    association_path,
)
from ..services.models import load_xgb, align_to_booster_features, forecast_regressor_next_days
import logging

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None
try:
    import xgboost as xgb
except Exception:
    xgb = None

router = APIRouter()
logger = logging.getLogger(__name__)


def _prob_band(p: float, n: int | None, z: float = 1.96):
    try:
        p = float(p)
    except Exception:
        return None
    N = int(n) if (n and int(n) > 0) else None
    if N is None or N <= 0:
        # fallback effective sample size
        N = 50
    se = (p * (1.0 - p) / max(1, N)) ** 0.5
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return {"lower": lo, "upper": hi, "n": N, "alpha": 0.05}


def _assess_bins(prob_pred, prob_true):
    """Return a list of one-word assessments for each bin: Good, Mixed, Over, Under."""
    out = []
    try:
        eps_good = 0.03
        eps_bad = 0.05
        for p, t in zip(prob_pred, prob_true):
            if p is None or t is None:
                out.append("Mixed")
                continue
            try:
                pdv = float(p)
                tdv = float(t)
            except Exception:
                out.append("Mixed")
                continue
            diff = pdv - tdv
            if abs(diff) <= eps_good:
                out.append("Good")
            elif diff > eps_bad:
                out.append("Over")
            elif diff < -eps_bad:
                out.append("Under")
            else:
                out.append("Mixed")
    except Exception:
        return []
    return out


def _safe_ticker(ticker: str | None) -> str:
    return (ticker or "AAPL").upper()


@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "title": "Home", "year": datetime.datetime.now().year},
    )


@router.get("/data", response_class=HTMLResponse)
def data_page(request: Request):
    return templates.TemplateResponse(
        "data.html",
        {"request": request, "title": "Data", "year": datetime.datetime.now().year},
    )


@router.get("/search", response_class=HTMLResponse)
def search_page(request: Request, ticker: str | None = None, period: str | None = None):
    ticker_info = None
    recent = None
    error = None
    # include popular equities and top crypto tickers (Yahoo Finance symbols)
    suggestions = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "JPM",
        "V",
        "NFLX",
        # crypto
        "BTC-USD",
        "ETH-USD",
        "USDT-USD",
        "BNB-USD",
        "XRP-USD",
        "USDC-USD",
        "SOL-USD",
    ]
    period_map = {
        "1w": "7d",
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y",
        "2y": "2y",
        "5y": "5y",
        "max": "max",
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
                "currency": (
                    getattr(info, "currency", None) if info else details.get("currency")
                ),
                "last_price": r3(
                    getattr(info, "last_price", None)
                    if info
                    else details.get("currentPrice")
                ),
                "previous_close": r3(
                    getattr(info, "previous_close", None)
                    if info
                    else details.get("previousClose")
                ),
                "year_high": r3(
                    getattr(info, "year_high", None)
                    if info
                    else details.get("fiftyTwoWeekHigh")
                ),
                "year_low": r3(
                    getattr(info, "year_low", None)
                    if info
                    else details.get("fiftyTwoWeekLow")
                ),
            }

            # Additional financial summary fields (may be absent depending on ticker/data source)
            try:
                market_cap = details.get("marketCap") if details else None
            except Exception:
                market_cap = None
            try:
                pe = (
                    details.get("trailingPE")
                    or details.get("trailingPegRatio")
                    or details.get("peRatio")
                )
            except Exception:
                pe = None
            try:
                div_yield = details.get("dividendYield")
            except Exception:
                div_yield = None
            try:
                beta = details.get("beta")
            except Exception:
                beta = None

            ticker_info.update(
                {
                    "market_cap": market_cap,
                    "pe_ratio": r3(pe) if pe is not None else None,
                    # dividendYield from yfinance is often 0.006 -> show as percent in template if desired
                    "dividend_yield": (
                        (
                            round(float(div_yield) * 100, 3)
                            if div_yield is not None
                            else None
                        )
                        if div_yield is not None
                        else None
                    ),
                    "beta": r3(beta) if beta is not None else None,
                }
            )

            # Company metadata for display
            try:
                sector = details.get("sector") if details else None
            except Exception:
                sector = None
            try:
                industry = details.get("industry") if details else None
            except Exception:
                industry = None
            try:
                country = details.get("country") if details else None
            except Exception:
                country = None
            try:
                website = details.get("website") if details else None
            except Exception:
                website = None
            try:
                summary = (
                    details.get("longBusinessSummary") or details.get("summary")
                    if details
                    else None
                )
            except Exception:
                summary = None
            try:
                employees = details.get("fullTimeEmployees") if details else None
            except Exception:
                employees = None

            ticker_info.update(
                {
                    "sector": sector,
                    "industry": industry,
                    "country": country,
                    "website": website,
                    "summary": summary,
                    "employees": employees,
                }
            )
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
def tasks_page(
    request: Request,
    ticker: str | None = None,
    ran: int | None = None,
    ts: str | None = None,
):
    t = _safe_ticker(ticker)
    return templates.TemplateResponse(
        "tasks.html",
        {
            "request": request,
            "title": "Tasks",
            "year": datetime.datetime.now().year,
            "ticker": t,
            "ran": bool(ran),
            "ts": ts,
        },
    )


# Roadmap page removed per request


@router.get("/contact", response_class=HTMLResponse)
def contact_page(request: Request):
    return templates.TemplateResponse(
        "contact.html",
        {"request": request, "title": "Support", "year": datetime.datetime.now().year},
    )


@router.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    # Show the upload page and, if a ticker is provided or a registry entry exists,
    # predict the next week (7 business days) and render results.
    t = _safe_ticker(request.query_params.get("ticker"))
    ts = request.query_params.get("ts")
    # Try loading model and features; if unavailable, render page without predictions
    try:
        booster, feat_names = load_xgb(t)
    except Exception:
        booster, feat_names = None, None

    # attempt to load model metadata for summary card
    model_meta = None
    try:
        model_path, model_meta_path = xgb_paths(t)
        if os.path.exists(model_meta_path):
            with open(model_meta_path, "r") as mf:
                model_meta = json.load(mf)
    except Exception:
        model_meta = None
    # attempt to load model metadata for summary card
    model_meta = None
    try:
        model_path, model_meta_path = xgb_paths(t)
        if os.path.exists(model_meta_path):
            with open(model_meta_path, "r") as mf:
                model_meta = json.load(mf)
    except Exception:
        model_meta = None

    if booster is None or not feat_names:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "title": "Upload",
                "year": datetime.datetime.now().year,
                "ticker": t,
                "ts": ts,
                "model_meta": model_meta,
            },
        )

    feat_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet"
    )
    if not os.path.exists(feat_path):
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "title": "Upload",
                "year": datetime.datetime.now().year,
                "ticker": t,
                "ts": ts,
                "model_meta": model_meta,
            },
        )

    try:
        df = pd.read_parquet(feat_path).reset_index(drop=True)
    except Exception:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "title": "Upload",
                "year": datetime.datetime.now().year,
                "ticker": t,
                "ts": ts,
                "model_meta": model_meta,
            },
        )

    # sort by date if present
    try:
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
    except Exception:
        pass

    try:
        dates, preds = forecast_regressor_next_days(df=df, booster=booster, feat_names=feat_names, days=7)
    except HTTPException as e:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "title": "Upload",
                "year": datetime.datetime.now().year,
                "error": e.detail,
                "ticker": t,
                "ts": ts,
                "model_meta": model_meta,
            },
        )
    except Exception:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "title": "Upload",
                "year": datetime.datetime.now().year,
                "error": "Failed to compute next-7-day predictions.",
                "ticker": t,
                "ts": ts,
                "model_meta": model_meta,
            },
        )

    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "title": "Upload",
            "year": datetime.datetime.now().year,
            "predictions": [float(x) for x in preds],
            "count": int(len(preds)),
            "dates": dates,
            "ticker": t,
            "ts": ts,
            "model_meta": model_meta,
        },
    )


@router.post("/upload", response_class=HTMLResponse)
def upload_predict(
    request: Request, ticker: str | None = Form("AAPL"), ts: str | None = Form(None)
):
    t = _safe_ticker(ticker)
    # load regressor and feature names
    try:
        booster, feat_names = load_xgb(t)
    except Exception:
        booster, feat_names = None, None

    # ensure model_meta is defined for template rendering
    model_meta = None
    try:
        model_path, model_meta_path = xgb_paths(t)
        if os.path.exists(model_meta_path):
            with open(model_meta_path, "r") as mf:
                model_meta = json.load(mf)
    except Exception:
        model_meta = None

    if booster is None or not feat_names:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "title": "Upload",
                "year": datetime.datetime.now().year,
                "error": "XGB model not available. Train it first.",
                "ticker": t,
                "ts": ts,
                "model_meta": model_meta,
            },
        )

    feat_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet"
    )
    if not os.path.exists(feat_path):
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "title": "Upload",
                "year": datetime.datetime.now().year,
                "error": f"Features file missing for {t}. Run the pipeline first.",
                "ticker": t,
                "ts": ts,
                "model_meta": model_meta,
            },
        )

    try:
        df = pd.read_parquet(feat_path).reset_index(drop=True)
    except Exception:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "title": "Upload",
                "year": datetime.datetime.now().year,
                "error": "Could not read feature parquet for ticker.",
                "ticker": t,
                "ts": ts,
                "model_meta": model_meta,
            },
        )

    # sort by date if present
    try:
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
    except Exception:
        pass

    try:
        dates, preds = forecast_regressor_next_days(df=df, booster=booster, feat_names=feat_names, days=7)
    except HTTPException as e:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "title": "Upload",
                "year": datetime.datetime.now().year,
                "error": e.detail,
                "ticker": t,
                "ts": ts,
                "model_meta": model_meta,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "title": "Upload",
                "year": datetime.datetime.now().year,
                "error": f"Prediction failed: {e}",
                "ticker": t,
                "ts": ts,
                "model_meta": model_meta,
            },
        )

    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "title": "Upload",
            "year": datetime.datetime.now().year,
            "predictions": [float(x) for x in preds],
            "count": int(len(preds)),
            "dates": dates,
            "ticker": t,
            "ts": ts,
            "model_meta": model_meta,
        },
    )


@router.get("/classify", response_class=HTMLResponse)
def classify_page(request: Request, ticker: str | None = None):
    t = _safe_ticker(ticker)
    _, cls_meta_path = xgb_classifier_paths(t)
    feats = []
    model_meta = None
    tokens = {}
    if os.path.exists(cls_meta_path):
        with open(cls_meta_path, "r") as f:
            meta = json.load(f)
        feats = [str(f) for f in meta.get("features", [])]
        model_meta = meta
        # build short tokens for UI (same logic as cluster/pca pages)
        try:
            for f in feats:
                parts = [p for p in re.split(r"[_\s\-]+", str(f)) if p]
                if parts:
                    tok = "".join([p[0] for p in parts]).upper()
                    tokens[f] = tok if len(tok) <= 6 else tok[:6]
                else:
                    tokens[f] = str(f)[:6]
        except Exception:
            tokens = {f: str(f)[:6] for f in feats}
    # attempt to locate pipeline-produced next-day prediction JSON
    prediction = None
    try:
        base = os.path.join(MODEL_REGISTRY, t)
        if os.path.isdir(base):
            candidates = [
                os.path.join(base, d)
                for d in os.listdir(base)
                if os.path.isdir(os.path.join(base, d))
            ]
            preds = []
            for c in candidates:
                p = os.path.join(c, f"predict_next_{t}.json")
                if os.path.exists(p):
                    preds.append((c, p))
            if preds:
                best = sorted(preds, key=lambda x: x[0])[-1]
                try:
                    with open(best[1], "r") as pf:
                        prediction = json.load(pf)
                except Exception:
                    prediction = None
    except Exception:
        prediction = None
    # fallback to flat registry
    if prediction is None:
        flat = os.path.join(MODEL_REGISTRY, f"predict_next_{t}.json")
        if os.path.exists(flat):
            try:
                with open(flat, "r") as pf:
                    prediction = json.load(pf)
            except Exception:
                prediction = None
    # compute probability band and calibration hints if we have a prediction
    calibration = None
    try:
        if (
            prediction
            and isinstance(prediction, dict)
            and prediction.get("proba_up") is not None
        ):
            samples = (
                model_meta.get("samples") if isinstance(model_meta, dict) else None
            )
            sample_n = None
            if isinstance(samples, dict):
                sample_n = (
                    samples.get("total") or samples.get("train") or samples.get("val")
                )
            else:
                sample_n = samples
            band = _prob_band(prediction.get("proba_up"), sample_n)
            metrics = (
                model_meta.get("metrics", {}) if isinstance(model_meta, dict) else {}
            )
            calibration = {
                "band": band,
                "auc": (
                    metrics.get("auc_val")
                    if metrics.get("auc_val") is not None
                    else metrics.get("auc")
                ),
                "logloss": (
                    metrics.get("logloss_val")
                    if metrics.get("logloss_val") is not None
                    else metrics.get("logloss")
                ),
            }
    except Exception:
        calibration = None

    # Prefer saved calibration from model metadata when available.
    reliability = None
    try:
        if isinstance(model_meta, dict):
            cal = model_meta.get("calibration")
            if isinstance(cal, dict) and cal.get("prob_true") and cal.get("prob_pred"):
                samples = model_meta.get("samples")
                n_val = None
                if isinstance(samples, dict):
                    n_val = samples.get("val") or samples.get("total")
                reliability = {
                    "prob_true": [
                        None if x is None else float(x)
                        for x in cal.get("prob_true", [])
                    ],
                    "prob_pred": [float(x) for x in cal.get("prob_pred", [])],
                    "brier": (
                        float(cal["brier"]) if cal.get("brier") is not None else None
                    ),
                    "n": int(n_val) if n_val is not None else None,
                    "source": "saved",
                }
                try:
                    logger.info(
                        "Using saved calibration for %s: n=%s bins=%s",
                        t,
                        reliability.get("n"),
                        len(reliability.get("prob_pred", [])),
                    )
                except Exception:
                    pass
                try:
                    rel_p = reliability.get("prob_pred", [])
                    rel_t = reliability.get("prob_true", [])
                    reliability["assess"] = _assess_bins(rel_p, rel_t)
                except Exception:
                    reliability["assess"] = []
    except Exception:
        reliability = None
    # compute reliability curve (calibration curve) using available features+labels if possible
    # (only if not already available from saved metadata)
    try:
        if reliability is not None:
            pass
        else:
            feat_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet"
            )
            if os.path.exists(feat_path):
                df_feats = pd.read_parquet(feat_path)
            # derive a simple next-day label similar to training: next-close > close
            if "log_return" in df_feats.columns:
                ret = df_feats["log_return"].shift(-1)
            elif "Close" in df_feats.columns:
                close = pd.to_numeric(df_feats["Close"], errors="coerce")
                ret = (close.shift(-1) - close) / close
            else:
                ret = None
            if ret is not None:
                valid = ~pd.isna(ret)
                dfx = df_feats.loc[valid].copy()
                y = (ret[valid] > 0).astype(int).to_numpy()

                # Build X based on the model's expected feature list (prevents feature mismatch).
                expected_feats = []
                try:
                    if isinstance(model_meta, dict) and model_meta.get("features"):
                        expected_feats = [
                            str(f) for f in model_meta.get("features", [])
                        ]
                except Exception:
                    expected_feats = []
                if expected_feats:
                    X = dfx.copy()
                    # ensure all expected columns exist
                    for col in expected_feats:
                        if col not in X.columns:
                            X[col] = np.nan
                    X = X[expected_feats]
                else:
                    X = dfx.select_dtypes(include=[np.number]).copy()

                X = X.replace([np.inf, -np.inf], np.nan)
                try:
                    X = X.ffill().bfill()
                except Exception:
                    pass
                for c in X.columns:
                    if X[c].isna().any():
                        try:
                            med = float(X[c].median(skipna=True))
                            if np.isnan(med):
                                med = 0.0
                        except Exception:
                            med = 0.0
                        X[c] = X[c].fillna(med)
                # try loading classifier and scoring
                try:
                    cls_model_path, _ = xgb_classifier_paths(t)
                    if xgb is not None and os.path.exists(cls_model_path):
                        booster = xgb.Booster()
                        booster.load_model(cls_model_path)
                        dmat = xgb.DMatrix(X)
                        probs = booster.predict(dmat)
                        # brier score
                        brier = None
                        try:
                            from sklearn.metrics import brier_score_loss

                            brier = float(brier_score_loss(y, probs))
                        except Exception:
                            brier = None
                        # calibration curve (sklearn) with fallback to manual binning
                        try:
                            from sklearn.calibration import calibration_curve

                            prob_true, prob_pred = calibration_curve(
                                y, probs, n_bins=10, strategy="uniform"
                            )
                        except Exception:
                            bins = np.linspace(0.0, 1.0, 11)
                            inds = np.digitize(probs, bins) - 1
                            prob_true = []
                            prob_pred = []
                            for i in range(10):
                                sel = inds == i
                                if sel.sum() == 0:
                                    prob_true.append(np.nan)
                                    prob_pred.append((bins[i] + bins[i + 1]) / 2.0)
                                else:
                                    prob_true.append(float(np.mean(y[sel])))
                                    prob_pred.append(float(np.mean(probs[sel])))
                            prob_true = np.array(prob_true)
                            prob_pred = np.array(prob_pred)
                        reliability = {
                            "prob_true": [
                                None if np.isnan(x) else float(x)
                                for x in prob_true.tolist()
                            ],
                            "prob_pred": [float(x) for x in prob_pred.tolist()],
                            "brier": brier,
                            "n": int(len(y)),
                            "source": "training",
                        }
                        try:
                            rel_p = reliability.get("prob_pred", [])
                            rel_t = reliability.get("prob_true", [])
                            reliability["assess"] = _assess_bins(rel_p, rel_t)
                        except Exception:
                            reliability["assess"] = []
                except Exception:
                    logger.exception("Runtime reliability compute failed for %s", t)
                    reliability = None
    except Exception:
        logger.exception("Reliability outer error for %s", t)
        reliability = None
    return templates.TemplateResponse(
        "classify.html",
        {
            "request": request,
            "title": "Classify",
            "year": datetime.datetime.now().year,
            "features": feats,
            "ticker": t,
            "prediction": prediction,
            "model_meta": model_meta,
            "calibration": calibration,
            "reliability": reliability,
            "tokens": tokens,
        },
    )


@router.post("/classify", response_class=HTMLResponse)
def classify_submit(
    request: Request, values: str = Form(...), ticker: str | None = None
):
    t = _safe_ticker(ticker)
    cls_model_path, cls_meta_path = xgb_classifier_paths(t)
    if xgb is None or not (
        os.path.exists(cls_meta_path) and os.path.exists(cls_model_path)
    ):
        return templates.TemplateResponse(
            "classify.html",
            {
                "request": request,
                "title": "Classify",
                "year": datetime.datetime.now().year,
                "features": [],
                "result": None,
                "ticker": t,
                "model_meta": None,
                "tokens": {},
            },
        )
    booster = xgb.Booster()
    booster.load_model(cls_model_path)
    with open(cls_meta_path, "r") as f:
        meta = json.load(f)
    feat_names = [str(f) for f in meta.get("features", [])]
    # build short tokens for UI (same logic as classify_page)
    tokens = {}
    try:
        for f in feat_names:
            parts = [p for p in re.split(r"[_\s\-]+", str(f)) if p]
            if parts:
                tok = "".join([p[0] for p in parts]).upper()
                tokens[f] = tok if len(tok) <= 6 else tok[:6]
            else:
                tokens[f] = str(f)[:6]
    except Exception:
        tokens = {f: str(f)[:6] for f in feat_names}
    try:
        nums = [float(x.strip()) for x in values.split(",") if x.strip() != ""]
    except Exception:
        return templates.TemplateResponse(
            "classify.html",
            {
                "request": request,
                "title": "Classify",
                "year": datetime.datetime.now().year,
                "features": feat_names,
                "result": None,
                "ticker": t,
                "tokens": tokens,
                "model_meta": meta,
            },
        )
    if len(nums) != len(feat_names):
        return templates.TemplateResponse(
            "classify.html",
            {
                "request": request,
                "title": "Classify",
                "year": datetime.datetime.now().year,
                "features": feat_names,
                "result": None,
                "ticker": t,
                "model_meta": meta,
                "tokens": tokens,
            },
        )
    df = pd.DataFrame([nums], columns=[f.strip() for f in feat_names])
    dmatrix = xgb.DMatrix(df)
    proba = float(booster.predict(dmatrix)[0])
    result = {"proba_up": proba, "label": int(proba >= 0.5)}
    return templates.TemplateResponse(
        "classify.html",
        {
            "request": request,
            "title": "Classify",
            "year": datetime.datetime.now().year,
            "features": feat_names,
            "result": result,
            "ticker": t,
            "model_meta": meta,
            "tokens": tokens,
        },
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
    # prepare feature tokens for UI (same style as cluster page)
    features = meta.get("feature_order", []) if isinstance(meta, dict) else []
    tokens = {}
    try:
        for f in features:
            parts = [p for p in re.split(r"[_\s\-]+", str(f)) if p]
            if parts:
                tok = "".join([p[0] for p in parts]).upper()
                tokens[f] = tok if len(tok) <= 6 else tok[:6]
            else:
                tokens[f] = str(f)[:6]
    except Exception:
        tokens = {f: str(f)[:6] for f in features}

    return templates.TemplateResponse(
        "pca.html",
        {
            "request": request,
            "title": "PCA",
            "year": datetime.datetime.now().year,
            "pca": meta,
            "error": error,
            "ticker": t,
            "features": features,
            "tokens": tokens,
        },
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
    # short tokens for UI chips (fallback to first 6 chars)
    tokens = {}
    try:
        for f in features:
            parts = [p for p in re.split(r"[_\s\-]+", str(f)) if p]
            if parts:
                tok = "".join([p[0] for p in parts]).upper()
                tokens[f] = tok if len(tok) <= 6 else tok[:6]
            else:
                tokens[f] = str(f)[:6]
    except Exception:
        tokens = {f: str(f)[:6] for f in features}
    # compute diagnostics for UI: how many rows in features file and how many used for training
    features_count = None
    training_count = None
    try:
        feat_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet"
        )
        if os.path.exists(feat_path):
            try:
                df_feats = pd.read_parquet(feat_path)
                features_count = int(len(df_feats))
            except Exception:
                features_count = None
    except Exception:
        features_count = None

    if isinstance(meta, dict):
        row_index = meta.get("row_index")
        if isinstance(row_index, (list, tuple)):
            training_count = int(len(row_index))
        else:
            # fallback: try labels file length
            try:
                _, labels_path = cluster_paths(t)
                if os.path.exists(labels_path):
                    labels = np.load(labels_path)
                    training_count = int(len(labels))
            except Exception:
                training_count = None

    # attach diagnostics into meta passed to template
    if isinstance(meta, dict):
        meta["features_count"] = features_count
        meta["training_count"] = training_count
    return templates.TemplateResponse(
        "cluster.html",
        {
            "request": request,
            "title": "Clustering",
            "year": datetime.datetime.now().year,
            "clusters": meta,
            "features": features,
            "tokens": tokens,
            "result": None,
            "error": error,
            "ticker": t,
        },
    )


@router.post("/cluster", response_class=HTMLResponse)
def cluster_submit(
    request: Request, values: str = Form(...), ticker: str | None = None
):
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
    # short tokens for UI chips (fallback to first 6 chars)
    tokens = {}
    try:
        for f in features:
            parts = [p for p in re.split(r"[_\s\-]+", str(f)) if p]
            if parts:
                tok = "".join([p[0] for p in parts]).upper()
                tokens[f] = tok if len(tok) <= 6 else tok[:6]
            else:
                tokens[f] = str(f)[:6]
    except Exception:
        tokens = {f: str(f)[:6] for f in features}
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
            if (
                isinstance(mean, list)
                and isinstance(scale, list)
                and len(mean) == len(x)
                and len(scale) == len(x)
            ):
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
            "tokens": tokens,
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
        {
            "request": request,
            "title": "Forecast",
            "year": datetime.datetime.now().year,
            "forecast": data,
            "error": error,
            "ticker": t,
        },
    )


@router.post("/forecast-page", response_class=HTMLResponse)
async def forecast_page_submit(
    request: Request, horizon: int = Form(7), ticker: str | None = Form(None)
):
    # Prefer the explicit form field, then query param, then Referer; fallback handled by _safe_ticker
    form = await request.form()
    form_ticker = form.get("ticker") if form is not None else None
    # try query param if form field missing
    query_ticker = request.query_params.get("ticker")
    # try Referer header query string as a last-ditch attempt
    referer = request.headers.get("referer") or request.headers.get("referrer")
    ref_ticker = None
    try:
        if referer and "?" in referer:
            qs = referer.split("?", 1)[1]
            for part in qs.split("&"):
                if part.startswith("ticker="):
                    ref_ticker = part.split("=", 1)[1]
                    break
    except Exception:
        ref_ticker = None
    t = _safe_ticker(form_ticker or ticker or query_ticker or ref_ticker)
    error = None
    result = None
    if ARIMA is None:
        error = (
            "statsmodels not installed on server; cannot compute on-demand forecast."
        )
    else:
        feature_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet"
        )
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
                    idx = pd.date_range(
                        last_date + pd.Timedelta(days=1),
                        periods=horizon,
                        freq=freq or "D",
                    )
                    result = {
                        "horizon": int(horizon),
                        "order": list(order),
                        "aic": float(getattr(fitted, "aic", float("nan"))),
                        "bic": float(getattr(fitted, "bic", float("nan"))),
                        "last_observation": float(series.iloc[-1]),
                        "dates": [d.isoformat() for d in idx],
                        "predictions": [float(x) for x in fc_vals.tolist()],
                        "confidence_interval": conf,
                    }
            except Exception as e:
                error = f"Forecast error: {e}"
    persisted = None
    FORECAST_META_PATH = forecast_path(t)
    # If we computed a new result successfully, persist it so the GET page
    # (which we redirect to) will display the newly generated forecast.
    if result is not None and error is None:
        # compute training diagnostics (log-likelihood, RMSE, MAE) from the fitted model
        log_likelihood = None
        rmse = None
        mae = None
        try:
            if "fitted" in locals() and hasattr(fitted, "llf"):
                log_likelihood = float(getattr(fitted, "llf", None))
        except Exception:
            log_likelihood = None
        try:
            if "fitted" in locals() and hasattr(fitted, "resid"):
                resid = np.asarray(getattr(fitted, "resid"))
                resid = resid[~np.isnan(resid)]
                if resid.size > 0:
                    rmse = float(np.sqrt(np.mean(resid**2)))
                    mae = float(np.mean(np.abs(resid)))
        except Exception:
            rmse = None
            mae = None

        out = {
            "name": "arima-forecast",
            "created_at": datetime.datetime.now().isoformat(),
            "horizon": int(result.get("horizon", 0)),
            "order": result.get("order", []),
            "aic": float(result.get("aic", float("nan"))),
            "bic": float(result.get("bic", float("nan"))),
            "last_observation": float(result.get("last_observation", float("nan"))),
            "predictions": result.get("predictions", []),
            "dates": result.get("dates", []),
            "confidence_interval": result.get("confidence_interval", []),
            "ticker": t,
            "log_likelihood": log_likelihood,
            "rmse": rmse,
            "mae": mae,
        }
        try:
            # ensure parent dir exists where forecast_path points
            parent = os.path.dirname(FORECAST_META_PATH)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(FORECAST_META_PATH, "w") as f:
                json.dump(out, f, indent=2)
            persisted = out
        except Exception:
            # persistence failure should not block redirect; fall through
            persisted = None
    else:
        if os.path.exists(FORECAST_META_PATH):
            try:
                with open(FORECAST_META_PATH, "r") as f:
                    persisted = json.load(f)
            except Exception:
                persisted = None

    # Redirect back to the GET page including the ticker query param so
    # the browser URL reflects the selected ticker (e.g. ?ticker=MSFT).
    return RedirectResponse(url=f"/forecast-page?ticker={t}", status_code=303)


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
    # prepare short tokens for UI (same style as cluster/pca pages)
    tokens = {}
    try:
        for f in features:
            parts = [p for p in re.split(r"[_\s\-]+", str(f)) if p]
            if parts:
                tok = "".join([p[0] for p in parts]).upper()
                tokens[f] = tok if len(tok) <= 6 else tok[:6]
            else:
                tokens[f] = str(f)[:6]
    except Exception:
        tokens = {f: str(f)[:6] for f in features}
    # prepare short tokens for UI (same style as cluster/pca pages)
    tokens = {}
    try:
        for f in features:
            parts = [p for p in re.split(r"[_\s\-]+", str(f)) if p]
            if parts:
                tok = "".join([p[0] for p in parts]).upper()
                tokens[f] = tok if len(tok) <= 6 else tok[:6]
            else:
                tokens[f] = str(f)[:6]
    except Exception:
        tokens = {f: str(f)[:6] for f in features}
    return templates.TemplateResponse(
        "recommend.html",
        {
            "request": request,
            "title": "Recommend",
            "year": datetime.datetime.now().year,
            "dates": dates,
            "features": features,
            "neighbors": None,
            "tokens": tokens,
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
                    neighbors = [
                        {"date": row_index[i], "distance": float(dists[i])}
                        for i in nn_idx
                    ]
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
                feat_path = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "ml",
                    "features",
                    f"{t}.parquet",
                )
                closes = {}
                next_returns_map = {}
                if os.path.exists(feat_path):
                    fdf = pd.read_parquet(feat_path)
                    fdf["date"] = pd.to_datetime(fdf["date"]).dt.strftime("%Y-%m-%d")
                    if "Close" in fdf.columns:
                        fdf["Close"] = pd.to_numeric(fdf["Close"], errors="coerce")
                        fdf["next_close"] = fdf["Close"].shift(-1)
                        fdf["next_return"] = (fdf["next_close"] - fdf["Close"]) / fdf[
                            "Close"
                        ]
                        closes = dict(
                            zip(
                                fdf["date"],
                                fdf.get("Close", pd.Series([None] * len(fdf))),
                            )
                        )
                        next_returns_map = dict(
                            zip(
                                fdf["date"],
                                fdf.get("next_return", pd.Series([None] * len(fdf))),
                            )
                        )
                    else:
                        closes = dict(zip(fdf["date"], pd.Series([None] * len(fdf))))
                neighbors = []
                for i in nn_idx:
                    dt = row_index[i]
                    dval = float(dists[i])
                    close_val = None
                    if closes:
                        c = closes.get(dt)
                        close_val = None if c is None else float(c)
                    nr = None
                    if next_returns_map:
                        v = next_returns_map.get(dt)
                        nr = (
                            None
                            if v is None or (isinstance(v, float) and (np.isnan(v)))
                            else float(v)
                        )
                    neighbors.append(
                        {
                            "date": dt,
                            "distance": dval,
                            "close": close_val,
                            "next_return": nr,
                        }
                    )
                # compute neighbor next-day return summary stats (if available)
                nr_vals = [
                    n.get("next_return")
                    for n in neighbors
                    if n.get("next_return") is not None
                ]
                if nr_vals:
                    avg_next = float(np.mean(nr_vals))
                    med_next = float(np.median(nr_vals))
                    std_next = float(np.std(nr_vals))
                else:
                    avg_next = med_next = std_next = None
    return templates.TemplateResponse(
        "recommend.html",
        {
            "request": request,
            "title": "Recommend",
            "year": datetime.datetime.now().year,
            "dates": dates,
            "features": features,
            "neighbors": neighbors,
            "avg_next_return": (avg_next if "avg_next" in locals() else None),
            "median_next_return": (med_next if "med_next" in locals() else None),
            "std_next_return": (std_next if "std_next" in locals() else None),
            "tokens": (locals().get("tokens") if "tokens" in locals() else {}),
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
    # gather provenance/metrics to show in UI
    assoc_meta = {}
    try:
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            assoc_meta["last_run"] = datetime.datetime.fromtimestamp(mtime).isoformat()
            assoc_meta["rules_path"] = f"/association-info?ticker={t}"
        # try to inspect features file for feature count and list
        feat_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet"
        )
        if os.path.exists(feat_path):
            try:
                df_feats = pd.read_parquet(feat_path)
                assoc_meta["feature_count"] = int(len(df_feats.columns))
                assoc_meta["feature_list"] = list(map(str, df_feats.columns[:50]))
            except Exception:
                assoc_meta["feature_count"] = None
                assoc_meta["feature_list"] = []
    except Exception:
        assoc_meta = {}

    return templates.TemplateResponse(
        "association.html",
        {
            "request": request,
            "title": "Association",
            "year": datetime.datetime.now().year,
            "assoc": data,
            "assoc_meta": assoc_meta,
            "error": error,
            "ticker": t,
        },
    )


# Selection persistence removed — association page is read-only
