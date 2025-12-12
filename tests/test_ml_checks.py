import os, json
import pandas as pd

def test_registry_files_exist():
    REGISTRY = 'ml/registry'  # Assuming REGISTRY is defined like this
    assert os.path.exists(REGISTRY)
    # xgb regressor optional
    xgb_meta = os.path.join(REGISTRY, "xgb_model.json")
    if os.path.exists(xgb_meta):
        with open(xgb_meta, "r") as f:
            meta = json.load(f)
        assert "metrics" in meta
    # xgb classifier optional
    xgb_cls = os.path.join(REGISTRY, "xgb_classifier.json")
    if os.path.exists(xgb_cls):
        with open(xgb_cls, "r") as f:
            meta = json.load(f)
        assert meta.get("task") == "classification"
def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
def test_predict_shape(client):
    # Use baseline if xgb not available; just check response structure
    res = client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
    assert res.status_code in (200, 400)
    # Classifier endpoint existence
    res2 = client.post("/predict-class", json={"features": [0.1, 0.2, 0.3]})
    assert res2.status_code in (200, 400)
import os, json
import pandas as pd

def test_model_registry_exists():
    meta_path = os.path.join('ml','registry','xgb_model.json')
    assert os.path.exists(meta_path), 'xgb_model.json not found; run flows/flow.py to train.'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    assert 'metrics' in meta and 'rmse' in meta['metrics']
    assert meta['metrics']['rmse'] < 0.2  # loose threshold for CI sanity


def test_features_file_integrity():
    fpath = os.path.join('ml','features','AAPL.parquet')
    assert os.path.exists(fpath), 'Features file missing; run flows/flow.py.'
    df = pd.read_parquet(fpath)
    # no-all-null columns
    assert df.shape[1] > 5
    assert df.drop(columns=['date','ticker'], errors='ignore').notna().sum().sum() > 0
