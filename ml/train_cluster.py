"""
Standalone clustering training script (mirrors Prefect flow cluster_features).
Reads a feature parquet/CSV, fits KMeans on numeric columns (excluding date/ticker),
and saves clusters_{ticker}.json + cluster_labels_{ticker}.npy to the registry.
"""

import argparse
import json
import os
from datetime import datetime, UTC

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception as e:  # pragma: no cover - missing dependency
    raise SystemExit("scikit-learn is required for clustering. Install scikit-learn.") from e


def load_features(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    return df


def fit_clusters(df: pd.DataFrame, n_clusters: int) -> tuple[dict, np.ndarray]:
    feature_cols = [
        c for c in df.columns if c not in ["date", "ticker"] and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    # capture the original row dates/indices used for training so we can align labels back
    row_index = None
    if 'date' in df.columns:
        try:
            row_index = df.loc[X.index, 'date'].astype(str).tolist()
        except Exception:
            row_index = [str(i) for i in X.index.tolist()]
    else:
        row_index = [str(i) for i in X.index.tolist()]
    if len(X) < n_clusters:
        raise ValueError("Insufficient rows to fit KMeans with the requested cluster count")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X_scaled)

    meta = {
        "name": "kmeans_clusters",
        "created_at": datetime.now(UTC).isoformat(),
        "n_clusters": n_clusters,
        "inertia": float(km.inertia_),
        "centers": km.cluster_centers_.tolist(),
        "feature_order": feature_cols,
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "row_index": row_index,
    }
    return meta, labels


def save_artifacts(meta: dict, labels: np.ndarray, registry_dir: str, ticker: str) -> tuple[str, str]:
    os.makedirs(registry_dir, exist_ok=True)
    meta_path = os.path.join(registry_dir, f"clusters_{ticker}.json")
    labels_path = os.path.join(registry_dir, f"cluster_labels_{ticker}.npy")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    np.save(labels_path, labels)
    return meta_path, labels_path


def main():
    parser = argparse.ArgumentParser(description="Train KMeans clusters from feature data")
    parser.add_argument(
        "--features",
        default=os.path.join(os.path.dirname(__file__), "features", "AAPL.parquet"),
        help="Path to feature parquet/CSV file",
    )
    parser.add_argument(
        "--registry",
        default=os.path.join(os.path.dirname(__file__), "registry"),
        help="Output registry directory",
    )
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")
    args = parser.parse_args()

    df = load_features(args.features)
    ticker = os.path.splitext(os.path.basename(args.features))[0].upper()
    meta, labels = fit_clusters(df, args.n_clusters)
    meta["ticker"] = ticker
    meta_path, labels_path = save_artifacts(meta, labels, args.registry, ticker)
    print(json.dumps({"status": "ok", "clusters_meta": meta_path, "cluster_labels": labels_path}, indent=2))


if __name__ == "__main__":
    main()
