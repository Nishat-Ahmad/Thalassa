#!/usr/bin/env python3
"""Run updated PCA training for all ticker feature files in ml/features."""
import datetime
import glob
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / 'ml' / 'features'
REGISTRY_DIR = ROOT / 'ml' / 'registry'

spec = importlib.util.spec_from_file_location('train_pca', str(ROOT / 'ml' / 'train_pca.py'))
train_pca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_pca)

files = glob.glob(str(FEATURE_DIR / '*.parquet')) + glob.glob(str(FEATURE_DIR / '*.csv'))
if not files:
    print('No feature files found in', FEATURE_DIR)
    raise SystemExit(1)

for f in files:
    print('\nProcessing', f)
    try:
        df = train_pca.load_features(f)
        ticker = os.path.splitext(os.path.basename(f))[0].upper()
        meta, comps = train_pca.fit_pca(df, n_components=5)
        meta['ticker'] = ticker
        out_dir = os.path.join(REGISTRY_DIR, ticker, datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S'))
        os.makedirs(out_dir, exist_ok=True)
        meta_path, comps_path = train_pca.save_artifacts(meta, comps, out_dir, ticker)
        # also save transformed info
        info = {'shape': list(comps.shape), 'dtype': str(comps.dtype), 'sample_min': float(np.nanmin(comps)), 'sample_max': float(np.nanmax(comps))}
        with open(os.path.join(out_dir, f'pca_transformed_{ticker}.json'), 'w') as jf:
            json.dump(info, jf, indent=2)
        print('Saved PCA for', ticker, '->', out_dir)
    except Exception as e:
        print('Failed for', f, 'error:', e)
