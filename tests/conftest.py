import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path so imports like `app` resolve during tests
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_ml_artifacts():
    """Create small synthetic feature file and minimal registry metadata so tests don't need network or trained models."""
    repo_root = Path(__file__).resolve().parents[1]
    ml_dir = repo_root / "ml"
    features_dir = ml_dir / "features"
    registry_dir = ml_dir / "registry"
    features_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    # If repository already contains AAPL features, prefer using them for realism/determinism
    repo_feat = features_dir / "AAPL.parquet"
    cwd = Path.cwd()
    cwd_ml = cwd / "ml"
    cwd_features = cwd_ml / "features"
    cwd_registry = cwd_ml / "registry"
    cwd_features.mkdir(parents=True, exist_ok=True)
    cwd_registry.mkdir(parents=True, exist_ok=True)

    if repo_feat.exists():
        # copy repo feature file into the working directory for tests
        try:
            with open(repo_feat, "rb") as src, open(cwd_features / "AAPL.parquet", "wb") as dst:
                dst.write(src.read())
            # read into df for metadata creation
            df = pd.read_parquet(repo_feat)
            wrote_parquet = True
        except Exception:
            df = None
            wrote_parquet = False
    else:
        df = None
        wrote_parquet = False

    # Create a small synthetic features dataframe if none available from repo
    if df is None:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=60, freq="D")
        close = (pd.Series(np.linspace(100.0, 120.0, num=len(dates))))
        df = pd.DataFrame({"date": dates, "Close": close, "Volume": np.random.randint(100, 1000, size=len(dates))})
    df["return"] = df["Close"].pct_change().fillna(0.0)
    df["log_return"] = np.log(df["Close"]).diff().fillna(0.0)
    for lag in [1, 2, 3]:
        df[f"lag_close_{lag}"] = df["Close"].shift(lag).fillna(method="bfill")

    # Prefer parquet; fallback to CSV if parquet engine not available in the environment
    feat_path = features_dir / "AAPL.parquet"
    try:
        # write back to repo features dir if it doesn't exist
        if not feat_path.exists():
            df.to_parquet(feat_path)
        # ensure working dir features copy exists
        if not (cwd_features / "AAPL.parquet").exists():
            with open(feat_path, "rb") as src, open(cwd_features / "AAPL.parquet", "wb") as dst:
                dst.write(src.read())
        wrote_parquet = True
    except Exception:
        # fallback to CSV
        try:
            csv_path = features_dir / "AAPL.csv"
            df.to_csv(csv_path, index=False)
            df.to_csv(cwd_features / "AAPL.csv", index=False)
            wrote_parquet = False
        except Exception:
            wrote_parquet = False

    # Minimal registry metadata to satisfy tests
    meta = {"metrics": {"rmse": 0.1}, "features": [str(c) for c in df.select_dtypes(include=[np.number]).columns]}
    # Prefer existing registry artifacts if present in repo; otherwise write minimal metadata
    repo_model = registry_dir / "xgb_model_AAPL.json"
    repo_cls = registry_dir / "xgb_classifier_AAPL.json"
    if repo_model.exists():
        try:
            with open(repo_model, "r") as src, open(cwd_registry / "xgb_model_AAPL.json", "w") as dst:
                dst.write(src.read())
        except Exception:
            pass
    else:
        with open(registry_dir / "xgb_model_AAPL.json", "w") as f:
            json.dump(meta, f)
        try:
            with open(cwd_registry / "xgb_model_AAPL.json", "w") as f:
                json.dump(meta, f)
        except Exception:
            pass

    if repo_cls.exists():
        try:
            with open(repo_cls, "r") as src, open(cwd_registry / "xgb_classifier_AAPL.json", "w") as dst:
                dst.write(src.read())
        except Exception:
            pass
    else:
        cls_meta = {"task": "classification", "features": [str(c) for c in df.select_dtypes(include=[np.number]).columns]}
        with open(registry_dir / "xgb_classifier_AAPL.json", "w") as f:
            json.dump(cls_meta, f)
        try:
            with open(cwd_registry / "xgb_classifier_AAPL.json", "w") as f:
                json.dump(cls_meta, f)
        except Exception:
            pass

    yield


@pytest.fixture()
def client():
    return TestClient(app)
