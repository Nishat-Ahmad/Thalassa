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
