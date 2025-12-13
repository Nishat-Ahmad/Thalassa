from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Request, Form
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import os, json, numpy as np, datetime, math
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
import hashlib

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


def _sanitize_for_json(obj):
    """Recursively replace non-finite floats (inf, -inf, nan) with None so JSON serialization succeeds."""
    # primitives
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float) or (hasattr(obj, 'dtype') and np.issubdtype(getattr(obj, 'dtype'), np.floating)):
        try:
            val = float(obj)
            return val if math.isfinite(val) else None
        except Exception:
            return None
    # numpy scalar types
    if isinstance(obj, (np.integer, np.floating)):
        try:
            val = obj.item()
            return val if (not isinstance(val, float) or math.isfinite(val)) else None
        except Exception:
            return None
    # dict or list/tuple
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    # fallback: try to convert to basic types
    try:
        return str(obj)
    except Exception:
        return None


def _collect_artifacts(ticker: str | None = None, run_dir: str | None = None):
    items = []

    # determine candidate run directory
    t = _safe_ticker(ticker)
    # ensure we have xgb paths available for extras even when run_dir is supplied
    xgb_model_path, xgb_meta_path = xgb_paths(t, run_dir=run_dir)
    if run_dir:
        candidate_dir = run_dir
    else:
        # try to find a run dir by seeing where xgb meta would live
        xgb_model_path, xgb_meta_path = xgb_paths(t)
        candidate_dir = os.path.dirname(xgb_meta_path) if xgb_meta_path else MODEL_REGISTRY
        if not os.path.isdir(candidate_dir) or (candidate_dir == MODEL_REGISTRY and not os.listdir(candidate_dir)):
            candidate_dir = MODEL_REGISTRY

        # if ticker subdir exists, prefer latest timestamped run directory
        ticker_base = os.path.join(MODEL_REGISTRY, t)
        if os.path.isdir(ticker_base):
            runs = [d for d in os.listdir(ticker_base) if os.path.isdir(os.path.join(ticker_base, d))]
            if runs:
                latest = sorted(runs)[-1]
                candidate_dir = os.path.join(ticker_base, latest)

    # walk candidate directory and detect artifacts
    try:
        files = sorted(os.listdir(candidate_dir))
    except Exception:
        files = []

    def _human_size(n: int) -> str:
        for unit in ['B','KB','MB','GB','TB']:
            if n < 1024.0:
                return f"{n:3.1f} {unit}"
            n /= 1024.0
        return f"{n:.1f}PB"

    for fn in files:
        path = os.path.join(candidate_dir, fn)
        entry = {"name": fn, "path": path, "status": "missing", "detail": None, "kind": None}
        if os.path.exists(path):
            entry["status"] = "ready"
            entry["kind"] = os.path.splitext(fn)[1].lstrip('.').lower()
            try:
                sz = os.path.getsize(path)
                entry["size_bytes"] = sz
                entry["size_human"] = _human_size(sz)
                entry["detail"] = entry.get("detail") or entry["size_human"]
                try:
                    entry_stat = os.stat(path)
                    entry["mtime"] = datetime.datetime.fromtimestamp(entry_stat.st_mtime).isoformat()
                except Exception:
                    entry["mtime"] = None
            except Exception:
                entry["detail"] = None

            # JSON metadata: try to load and extract lightweight summary
            if fn.endswith('.json'):
                try:
                    j = _load_json(path)
                    if isinstance(j, dict):
                        # add common summary keys
                        if 'features' in j:
                            entry['detail'] = f"{len(j.get('features', []))} features"
                        elif 'feature_order' in j:
                            entry['detail'] = f"{len(j.get('feature_order', []))} feature dims"
                        elif 'centers' in j:
                            entry['detail'] = f"{len(j.get('centers', []))} centers"
                        elif 'generated_at' in j:
                            entry['detail'] = f"generated {j.get('generated_at')}"
                        else:
                            # fallback to listing top-level keys
                            entry['detail'] = 'keys: ' + ','.join(list(j.keys())[:5])
                    else:
                        entry['detail'] = 'json'
                except Exception:
                    entry['detail'] = 'json (invalid)'

            # numpy arrays: report shape when possible
            if fn.endswith('.npy'):
                try:
                    # load in mmap mode to avoid memory pressure
                    arr = np.load(path, mmap_mode='r')
                    entry['npy_shape'] = getattr(arr, 'shape', None)
                    entry['npy_dtype'] = str(getattr(arr, 'dtype', None))
                    entry['detail'] = f"shape {entry['npy_shape']} dtype {entry['npy_dtype']}"
                    # small summary: min/max on first axis sample if large
                    try:
                        if getattr(arr, 'size', 0) and arr.size <= 2000000:
                            a = np.array(arr)
                            entry['npy_min'] = float(a.min()) if a.size else None
                            entry['npy_max'] = float(a.max()) if a.size else None
                        else:
                            # sample first 100 rows/elements if possible
                            s = arr.flat[:100]
                            entry['npy_sample_min'] = float(min(s)) if hasattr(s, '__iter__') else None
                            entry['npy_sample_max'] = float(max(s)) if hasattr(s, '__iter__') else None
                    except Exception:
                        pass
                except Exception:
                    pass

            # model binary formats: show filename and size
            if fn.endswith('.ubj') or fn.endswith('.bin') or fn.endswith('.model'):
                # keep size already set
                pass

        items.append(entry)

    # Also include known artifact paths (legacy locations) that may not be in the latest run dir
    extras = []
    # common known files to check at registry root
    known = [xgb_meta_path, xgb_model_path,]
    for p in known:
        if p and os.path.exists(p):
            bn = os.path.basename(p)
            if not any(it.get('name') == bn for it in items):
                try:
                    st = os.stat(p)
                    mtime_iso = datetime.datetime.fromtimestamp(st.st_mtime).isoformat()
                    sz = st.st_size
                except Exception:
                    mtime_iso = None
                    sz = 0
                extras.append({
                    "name": bn,
                    "path": p,
                    "status": "ready",
                    "detail": f"{round(sz/1024,1)} KB",
                    "kind": os.path.splitext(bn)[1].lstrip('.'),
                    "size_bytes": sz,
                    "size_human": _human_size(sz),
                    "mtime": mtime_iso,
                })

    return items + extras


def _collect_models_for_run(ticker: str | None = None, run_dir: str | None = None):
    """Return a list of model summaries (type, present, brief info) for the given run directory."""
    t = _safe_ticker(ticker)
    summaries = []
    # xgb regressor
    xgb_model_path, xgb_meta_path = xgb_paths(t, run_dir=run_dir)
    if os.path.exists(xgb_meta_path):
        jm = _load_json(xgb_meta_path) or {}
        summaries.append({
            "name": "XGB Regressor",
            "type": "xgb",
            "present": True,
            "meta": jm,
            "info": f"features={len(jm.get('features', []))}" if isinstance(jm, dict) and jm.get('features') else "meta"
        })
    # xgb classifier
    xgbc_model_path, xgbc_meta_path = xgb_classifier_paths(t, run_dir=run_dir)
    if os.path.exists(xgbc_meta_path):
        jm = _load_json(xgbc_meta_path) or {}
        summaries.append({
            "name": "XGB Classifier",
            "type": "xgb_classifier",
            "present": True,
            "meta": jm,
            "info": f"features={len(jm.get('features', []))}" if isinstance(jm, dict) and jm.get('features') else "meta"
        })
    # pca
    pca_meta_path, _ = pca_paths(t, run_dir=run_dir)
    if os.path.exists(pca_meta_path):
        jm = _load_json(pca_meta_path) or {}
        info = None
        if isinstance(jm, dict):
            if jm.get('components'):
                info = f"components={len(jm.get('components', []))}"
            elif jm.get('explained_variance'):
                info = f"explained={jm.get('explained_variance')[:3]}"
        summaries.append({"name": "PCA", "type": "pca", "present": True, "meta": jm, "info": info or 'meta'})
    # clustering
    cluster_meta_path, _ = cluster_paths(t, run_dir=run_dir)
    if os.path.exists(cluster_meta_path):
        jm = _load_json(cluster_meta_path) or {}
        info = f"centers={len(jm.get('centers', []))}" if isinstance(jm, dict) and jm.get('centers') else 'meta'
        summaries.append({"name": "Clustering", "type": "clustering", "present": True, "meta": jm, "info": info})
    # forecast
    forecast_meta_path = forecast_path(t, run_dir=run_dir)
    if os.path.exists(forecast_meta_path):
        jm = _load_json(forecast_meta_path) or {}
        info = 'forecast' if jm else 'meta'
        summaries.append({"name": "Forecast", "type": "forecast", "present": True, "meta": jm, "info": info})
    # association
    assoc_meta_path = association_path(t, run_dir=run_dir)
    if os.path.exists(assoc_meta_path):
        jm = _load_json(assoc_meta_path) or {}
        info = 'rules' if jm else 'meta'
        summaries.append({"name": "Association Rules", "type": "association", "present": True, "meta": jm, "info": info})

    return summaries

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
def model_info(request: Request, ticker: str = Query("AAPL"), raw: str | None = Query(None), run: str | None = Query(None)):
    # if ticker == 'ALL' render a registry-wide overview
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

    # enumerate all run directories for this ticker (if any) and collect artifacts per run
    runs_data = []
    ticker_base = os.path.join(MODEL_REGISTRY, t)
    if os.path.isdir(ticker_base):
        runs = sorted([d for d in os.listdir(ticker_base) if os.path.isdir(os.path.join(ticker_base, d))])
        for r in runs:
            rd = os.path.join(ticker_base, r)
            try:
                art = _collect_artifacts(t, run_dir=rd)
            except Exception:
                art = []
            # collect model summaries for this run
            try:
                models = _collect_models_for_run(t, run_dir=rd)
            except Exception:
                models = []
            runs_data.append({"run": r, "path": rd, "artifacts": art, "models": models})

    # If user requested a full registry overview (ticker=ALL), scan every ticker folder
    registry_overview = None
    if ticker and str(ticker).upper() in ("ALL", "*"):
        registry_overview = []
        try:
            for tk in sorted([d for d in os.listdir(MODEL_REGISTRY) if os.path.isdir(os.path.join(MODEL_REGISTRY, d))]):
                tbase = os.path.join(MODEL_REGISTRY, tk)
                truns = sorted([d for d in os.listdir(tbase) if os.path.isdir(os.path.join(tbase, d))])
                ticker_entry = {"ticker": tk, "runs": []}
                for r in truns:
                    rd = os.path.join(tbase, r)
                    try:
                        art = _collect_artifacts(tk, run_dir=rd)
                    except Exception:
                        art = []
                    try:
                        models = _collect_models_for_run(tk, run_dir=rd)
                    except Exception:
                        models = []
                    ticker_entry["runs"].append({"run": r, "path": rd, "artifacts": art, "models": models})
                registry_overview.append(ticker_entry)
        except Exception:
            registry_overview = registry_overview or []

    # If client asked for a specific artifact's raw content, return it as JSON (safe, registry-only)
    if raw:
        # if a run is provided, look inside that run directory first
        safe_name = os.path.basename(raw)
        if run:
            candidate_dir = os.path.join(MODEL_REGISTRY, t, run)
            target = os.path.join(candidate_dir, safe_name)
        else:
            # determine candidate run dir (reuse similar logic from _collect_artifacts)
            candidate_dir = os.path.dirname(xgb_meta_path) if xgb_meta_path else MODEL_REGISTRY
            ticker_base = os.path.join(MODEL_REGISTRY, t)
            if os.path.isdir(ticker_base):
                runs = [d for d in os.listdir(ticker_base) if os.path.isdir(os.path.join(ticker_base, d))]
                if runs:
                    latest = sorted(runs)[-1]
                    candidate_dir = os.path.join(ticker_base, latest)
            target = os.path.join(candidate_dir, safe_name)
        if not os.path.exists(target):
            # fallback to registry root
            target = os.path.join(MODEL_REGISTRY, safe_name)

        if not os.path.exists(target):
            raise HTTPException(status_code=404, detail="Artifact not found")

        info = {}
        try:
            st = os.stat(target)
            info['size_bytes'] = st.st_size
            info['mtime'] = datetime.datetime.fromtimestamp(st.st_mtime).isoformat()
        except Exception:
            pass

        if target.endswith('.json'):
            content = _load_json(target)
            return {"meta": info, "content": _sanitize_for_json(content)}
        if target.endswith('.npy'):
            try:
                arr = np.load(target, mmap_mode='r')
                out = {"shape": getattr(arr, 'shape', None), "dtype": str(getattr(arr, 'dtype', None))}
                # attempt small sample summary
                try:
                    sample = np.array(arr.flat[:200])
                    out.update({"sample_min": float(sample.min()), "sample_max": float(sample.max())})
                except Exception:
                    pass
                return {"meta": info, "content": _sanitize_for_json(out)}
            except Exception:
                raise HTTPException(status_code=500, detail="Could not read npy")
        # fallback: return basic file info
        try:
            return {"meta": info, "name": os.path.basename(target)}
        except Exception:
            raise HTTPException(status_code=500, detail="Cannot read artifact")

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
        # build registry tickers list for dropdown selector
        registry_tickers = []
        try:
            registry_tickers = sorted([d for d in os.listdir(MODEL_REGISTRY) if os.path.isdir(os.path.join(MODEL_REGISTRY, d))])
        except Exception:
            registry_tickers = []

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
                "runs": runs_data,
                "registry": registry_overview,
                "registry_tickers": registry_tickers,
            },
        )

    return legacy_meta


@router.get("/cluster-samples")
def cluster_samples(
    ticker: str = Query("AAPL"),
    n: int = Query(3, gt=0, le=20),
    since: str | None = Query(None),
    until: str | None = Query(None),
):
    """Return up to `n` representative training rows per cluster for a ticker.

    This helps users inspect what each cluster looks like in raw feature space.
    """
    t = _safe_ticker(ticker)
    meta_path, labels_path = cluster_paths(t)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Cluster metadata not found")
    if not os.path.exists(labels_path):
        raise HTTPException(status_code=404, detail="Cluster labels not found")
    meta = _load_json(meta_path) or {}
    try:
        labels = np.load(labels_path)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not load cluster labels")

    feat_path = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet")
    if not os.path.exists(feat_path):
        raise HTTPException(status_code=404, detail="Feature file not found for ticker")
    try:
        df = pd.read_parquet(feat_path)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not read feature parquet for ticker")

    # normalize date column and apply optional date range filtering (since/until are YYYY-MM-DD)
    if "date" in df.columns:
        try:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if since:
                try:
                    since_dt = pd.to_datetime(since)
                    df = df[df["date"] >= since_dt]
                except Exception:
                    pass
            if until:
                try:
                    until_dt = pd.to_datetime(until)
                    df = df[df["date"] <= until_dt]
                except Exception:
                    pass
        except Exception:
            # if parsing fails, continue without filtering
            pass

    # Prefer aligning labels by saved row index (dates or original indices) when available.
    row_index = meta.get("row_index")
    df_used = None
    if row_index and isinstance(row_index, (list, tuple)):
        try:
            # try matching by date column first
            if "date" in df.columns:
                df = df.copy()
                df["__date_str"] = df["date"].astype(str)
                labels_df = pd.DataFrame({"__date_str": [str(x) for x in row_index], "cluster_label": labels})
                df_used = pd.merge(df, labels_df, on="__date_str", how="inner").drop(columns=["__date_str"])\
                    .reset_index(drop=True)
            else:
                # fallback: match on original index string values
                df_idx = df.reset_index()
                df_idx["__idx_str"] = df_idx["index"].astype(str)
                labels_df = pd.DataFrame({"__idx_str": [str(x) for x in row_index], "cluster_label": labels})
                df_used = pd.merge(df_idx, labels_df, on="__idx_str", how="inner").drop(columns=["__idx_str", "index"])\
                    .reset_index(drop=True)
        except Exception:
            df_used = None

    # fallback: align to the tail if we couldn't align by saved row index
    if df_used is None:
        minlen = min(len(labels), len(df))
        if minlen == 0:
            return {"ticker": t, "samples": {}}
        df_used = df.tail(minlen).reset_index(drop=True).copy()
        labels_trim = labels[-minlen:]
        df_used["cluster_label"] = labels_trim

    # Prioritize newest rows: if date column exists, sort by date desc; otherwise reverse index order
    try:
        if "date" in df_used.columns:
            df_used["date"] = pd.to_datetime(df_used["date"], errors="coerce")
            df_used = df_used.sort_values("date", ascending=False).reset_index(drop=True)
        else:
            df_used = df_used.sort_index(ascending=False).reset_index(drop=True)
    except Exception:
        # ignore sorting errors and proceed
        pass

    n_clusters = int(meta.get("n_clusters", int(labels.max()) + 1))
    out = {}
    for cid in range(n_clusters):
        sel = df_used[df_used["cluster_label"] == cid].head(n)
        out[str(cid)] = sel.to_dict(orient="records")

    # include the cluster metadata (centers, feature_order, scaler params) so front-end
    # visualizations (heatmap / violin) can render without an extra request
    resp_meta = _sanitize_for_json(meta.copy() if isinstance(meta, dict) else meta) or {}
    resp_meta["n_clusters"] = n_clusters

    return {"ticker": t, "meta": resp_meta, "samples": out}


@router.post("/label-dataset")
def label_dataset(ticker: str = Query("AAPL")):
    """Attach cluster labels to the feature dataset and save a labeled copy into a new run directory.

    Returns the path of the saved labeled dataset (parquet if possible, CSV fallback).
    """
    t = _safe_ticker(ticker)
    meta_path, labels_path = cluster_paths(t)
    if not os.path.exists(meta_path) or not os.path.exists(labels_path):
        raise HTTPException(status_code=404, detail="Cluster artifacts not found")
    try:
        labels = np.load(labels_path)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not load cluster labels")

    feat_path = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet")
    if not os.path.exists(feat_path):
        raise HTTPException(status_code=404, detail="Feature file not found for ticker")
    try:
        df = pd.read_parquet(feat_path)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not read feature parquet for ticker")

    minlen = min(len(labels), len(df))
    if minlen == 0:
        raise HTTPException(status_code=400, detail="No rows available to label")

    # attach labels to the rows using saved row_index when possible
    meta = _load_json(meta_path) or {}
    row_index = meta.get("row_index")
    if row_index and isinstance(row_index, (list, tuple)):
        try:
            if "date" in df.columns:
                df["date_str_for_label"] = df["date"].astype(str)
                labels_df = pd.DataFrame({"date_str_for_label": [str(x) for x in row_index], "cluster_label": labels})
                df_to_label = pd.merge(df, labels_df, on="date_str_for_label", how="inner").drop(columns=["date_str_for_label"]).reset_index(drop=True)
            else:
                df_idx = df.reset_index()
                df_idx["idx_str_for_label"] = df_idx["index"].astype(str)
                labels_df = pd.DataFrame({"idx_str_for_label": [str(x) for x in row_index], "cluster_label": labels})
                df_to_label = pd.merge(df_idx, labels_df, on="idx_str_for_label", how="inner").drop(columns=["idx_str_for_label", "index"]).reset_index(drop=True)
        except Exception:
            df_to_label = df.tail(minlen).reset_index(drop=True).copy()
            df_to_label["cluster_label"] = labels[-minlen:]
    else:
        # fallback: attach labels to the tail portion corresponding to training rows
        df_to_label = df.tail(minlen).reset_index(drop=True).copy()
        df_to_label["cluster_label"] = labels[-minlen:]

    # create a new run directory to store the labeled dataset
    try:
        from ..core import ensure_run_dir
    except Exception:
        raise HTTPException(status_code=500, detail="Cannot access run dir helper")
    run_dir = ensure_run_dir(t)
    out_parquet = os.path.join(run_dir, f"features_labeled_{t}.parquet")
    out_csv = os.path.join(run_dir, f"features_labeled_{t}.csv")
    try:
        df_to_label.to_parquet(out_parquet)
        saved = out_parquet
    except Exception:
        try:
            df_to_label.to_csv(out_csv, index=False)
            saved = out_csv
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to save labeled dataset")

    return {"status": "ok", "saved_path": saved}


@router.get("/cluster-interpret")
def cluster_interpret(ticker: str = Query("AAPL")):
    """Return simple semantic labels for clusters using center z-values (centers are in scaled space).

    This uses heuristic rules on common features (return, vol_10, rsi_14, macd, macd_signal)
    and a top-features list (largest absolute z-values) as reasoning.
    """
    t = _safe_ticker(ticker)
    meta_path, _ = cluster_paths(t)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Cluster metadata not found")
    meta = _load_json(meta_path) or {}
    centers = np.array(meta.get("centers", []), dtype=float)
    feat_order = meta.get("feature_order", []) or []
    if centers.size == 0 or not feat_order:
        raise HTTPException(status_code=400, detail="Cluster metadata incomplete")

    # Try to convert centers from scaled space back to original units when scaler params exist
    scaler_mean = np.array(meta.get("scaler_mean", []) if meta.get("scaler_mean") is not None else [])
    scaler_scale = np.array(meta.get("scaler_scale", []) if meta.get("scaler_scale") is not None else [])
    use_orig = False
    centers_orig = centers.copy()
    if scaler_mean.size and scaler_scale.size and centers.shape[1] == scaler_mean.size == scaler_scale.size:
        try:
            centers_orig = centers * scaler_scale + scaler_mean
            use_orig = True
        except Exception:
            centers_orig = centers.copy()

    # For each feature compute cluster percentile rank across centers (0..1)
    k = centers_orig.shape[0]
    feat_vals = {}
    for j, feat in enumerate(feat_order):
        vals = centers_orig[:, j]
        # compute percentile rank for each center value
        if k > 1:
            order = np.argsort(vals)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(k)
            ranks = ranks / float(k - 1)
        else:
            ranks = np.array([0.5])
        feat_vals[feat] = {"vals": vals, "ranks": ranks}

    interpretations = {}
    for idx in range(k):
        label_parts = []
        reasons = []

        def finfo(name):
            if name not in feat_vals:
                return None, None
            return float(feat_vals[name]["vals"][idx]), float(feat_vals[name]["ranks"][idx])

        # Heuristic thresholds (percentiles)
        high_thr = 0.66
        low_thr = 0.33

        # Returns
        for ret_name in ("return", "log_return"):
            v, p = finfo(ret_name)
            if v is None:
                continue
            if p >= high_thr:
                label_parts.append("Positive Return")
                reasons.append(f"{ret_name} p{p:.2f} val={v:.4g}{' (orig)' if use_orig else ''}")
            elif p <= low_thr:
                label_parts.append("Negative Return")
                reasons.append(f"{ret_name} p{p:.2f} val={v:.4g}{' (orig)' if use_orig else ''}")
            break

        # Volatility
        v, p = finfo("vol_10")
        if v is not None:
            if p >= high_thr:
                label_parts.append("High Volatility")
                reasons.append(f"vol_10 p{p:.2f} val={v:.4g}")
            elif p <= low_thr:
                label_parts.append("Low Volatility")
                reasons.append(f"vol_10 p{p:.2f} val={v:.4g}")

        # RSI
        v, p = finfo("rsi_14")
        if v is not None:
            if p >= high_thr:
                label_parts.append("Overbought")
                reasons.append(f"rsi_14 p{p:.2f} val={v:.4g}")
            elif p <= low_thr:
                label_parts.append("Oversold")
                reasons.append(f"rsi_14 p{p:.2f} val={v:.4g}")

        # MACD
        v, p = finfo("macd")
        if v is not None:
            if p >= high_thr:
                label_parts.append("Bullish MACD")
                reasons.append(f"macd p{p:.2f} val={v:.4g}")
            elif p <= low_thr:
                label_parts.append("Bearish MACD")
                reasons.append(f"macd p{p:.2f} val={v:.4g}")

        # Select top features by absolute deviation from median across centers (original units if available)
        top_idxs = np.argsort(np.abs(centers_orig[:, :] - np.median(centers_orig, axis=0))[idx, :])[::-1][:4]
        top_feats = []
        for j in top_idxs:
            if j < len(feat_order):
                top_feats.append({"feature": feat_order[j], "value": float(centers_orig[idx, j]), "percentile": float(feat_vals[feat_order[j]]["ranks"][idx])})

        if not label_parts:
            label = "Neutral"
        else:
            # deduplicate while preserving order
            seen = []
            for part in label_parts:
                if part not in seen:
                    seen.append(part)
            label = " + ".join(seen)

        interpretations[str(idx)] = {"label": label, "reason": reasons, "top_features": top_feats}

    return {"ticker": t, "interpretations": interpretations, "units": ("original" if use_orig else "scaled")}

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

    # attach latest label if available
    labels = None
    if os.path.exists(labels_path):
        try:
            labels = np.load(labels_path)
            meta["latest_label"] = int(labels[-1]) if len(labels) else None
        except Exception:
            meta["latest_label"] = None

    # diagnostic: how many feature rows exist and how many were used for training
    feat_path = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "features", f"{t}.parquet")
    features_count = None
    try:
        if os.path.exists(feat_path):
            try:
                df = pd.read_parquet(feat_path)
                features_count = int(len(df))
            except Exception:
                features_count = None
    except Exception:
        features_count = None

    # training rows count: prefer explicit row_index saved in meta, fallback to labels length
    training_count = None
    if isinstance(meta.get("row_index"), (list, tuple)):
        training_count = int(len(meta.get("row_index") or []))
    elif labels is not None:
        training_count = int(len(labels))

    meta["features_count"] = features_count
    meta["training_count"] = training_count
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


@router.get("/artifact-download")
def artifact_download(raw: str = Query(...), ticker: str = Query("AAPL")):
    # Return a file response for a given artifact name if it exists in latest run dir or registry root
    t = _safe_ticker(ticker)
    xgb_model_path, xgb_meta_path = xgb_paths(t)
    candidate_dir = os.path.dirname(xgb_meta_path) if xgb_meta_path else MODEL_REGISTRY
    ticker_base = os.path.join(MODEL_REGISTRY, t)
    if os.path.isdir(ticker_base):
        runs = [d for d in os.listdir(ticker_base) if os.path.isdir(os.path.join(ticker_base, d))]
        if runs:
            latest = sorted(runs)[-1]
            candidate_dir = os.path.join(ticker_base, latest)

    safe_name = os.path.basename(raw)
    target = os.path.join(candidate_dir, safe_name)
    if not os.path.exists(target):
        target = os.path.join(MODEL_REGISTRY, safe_name)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Artifact not found")

    from fastapi.responses import FileResponse
    return FileResponse(path=target, filename=safe_name, media_type='application/octet-stream')
