#!/usr/bin/env python3
"""Apply primary PCA fixes for BAC: log1p Volume, drop constant cols, StandardScaler, PCA(n_components=5).
Saves pca_BAC.json and pca_transformed_BAC.npy into a new timestamped registry folder.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except Exception:
    print("Missing sklearn. Install with: pip install scikit-learn")
    raise

ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / 'ml' / 'features' / 'BAC.parquet'
REGISTRY_BASE = ROOT / 'ml' / 'registry' / 'BAC'

if not FEATURE_PATH.exists():
    raise SystemExit(f"Features file not found: {FEATURE_PATH}")

print('Loading features from', FEATURE_PATH)
df = pd.read_parquet(FEATURE_PATH)
print('Original shape:', df.shape)

# Keep index (row_index) as strings if present
row_index = None
try:
    if hasattr(df.index, 'astype'):
        row_index = [str(i) for i in df.index.tolist()]
except Exception:
    row_index = None

# Drop any constant or near-constant columns (zero variance)
nunique = df.nunique(dropna=False)
keep_cols = [c for c in df.columns if nunique.get(c,0) > 1]
removed = [c for c in df.columns if c not in keep_cols]
if removed:
    print('Dropping constant/near-constant columns:', removed)
df = df[keep_cols]

# Ensure 'Volume' exists and apply log1p
if 'Volume' in df.columns:
    # convert to numeric if needed
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0.0)
    df['Volume'] = np.log1p(df['Volume'])
    print('Applied log1p to Volume')
else:
    print('Volume column not present; skipping log1p')

# Select numeric columns only
num_df = df.select_dtypes(include=[np.number]).copy()
print('Numeric feature shape after pruning:', num_df.shape)
feature_order = list(num_df.columns)

# Standardize
scaler = StandardScaler()
Xs = scaler.fit_transform(num_df.values)

# PCA
n_components = 5
pca = PCA(n_components=n_components)
Xp = pca.fit_transform(Xs)

# Prepare registry folder
ts = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
out_dir = REGISTRY_BASE / ts
out_dir.mkdir(parents=True, exist_ok=True)

# Save transformed array
trans_path = out_dir / 'pca_transformed_BAC.npy'
np.save(trans_path, Xp)
print('Saved transformed array to', trans_path)

# Save metadata
meta = {
    'name': 'pca_features',
    'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
    'n_components': n_components,
    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
    'components': pca.components_.tolist(),
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_order': feature_order,
    'row_index': row_index,
    'ticker': 'BAC'
}
with open(out_dir / 'pca_BAC.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('Saved PCA metadata to', out_dir / 'pca_BAC.json')

# Save a small summary JSON about the transformed array
trans_info = {
    'shape': list(Xp.shape),
    'dtype': str(Xp.dtype),
    'sample_min': float(np.nanmin(Xp)) if Xp.size else None,
    'sample_max': float(np.nanmax(Xp)) if Xp.size else None
}
with open(out_dir / 'pca_transformed_BAC.json', 'w') as f:
    json.dump(trans_info, f, indent=2)
print('Saved transformed info to', out_dir / 'pca_transformed_BAC.json')

print('\nExplained variance ratio:')
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f'  PC{i+1}: {v:.6f}')

print('\nDone. New registry folder:', out_dir)
