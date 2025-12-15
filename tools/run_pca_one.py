#!/usr/bin/env python3
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec_path = ROOT / 'ml' / 'train_pca.py'
spec = importlib.util.spec_from_file_location('train_pca', str(spec_path))
train_pca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_pca)

ticker = sys.argv[1] if len(sys.argv)>1 else 'GOOGL'
feature_path = ROOT / 'ml' / 'features' / f'{ticker}.parquet'
if not feature_path.exists():
    print('Features not found:', feature_path)
    sys.exit(1)
print('Running PCA for', ticker, 'from', feature_path)
df = train_pca.load_features(str(feature_path))
meta, comps = train_pca.fit_pca(df, n_components=5)
meta['ticker'] = ticker
out_dir = ROOT / 'ml' / 'registry' / ticker / datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
out_dir.mkdir(parents=True, exist_ok=True)
meta_path = out_dir / f'pca_{ticker}.json'
comps_path = out_dir / f'pca_transformed_{ticker}.npy'
with open(meta_path,'w') as f:
    json.dump(meta,f,indent=2)
np.save(comps_path, comps)
print('Saved to', out_dir)
print('Explained variance ratio:')
for i,v in enumerate(meta['explained_variance_ratio']):
    print(f'  PC{i+1}: {v:.6f}')
