"""
Standalone PCA training script to mirror the Prefect flow's PCA step.
Reads a feature parquet/CSV, fits PCA on numeric columns, and saves artifacts
(pca_{ticker}.json + pca_transformed_{ticker}.npy) to the registry directory.
"""

import argparse
import json
import os
from datetime import datetime, UTC

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception as e:  # pragma: no cover - missing dependency
    raise SystemExit("sklearn is required for PCA. Install scikit-learn.") from e


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


def fit_pca(df: pd.DataFrame, n_components: int) -> tuple[dict, np.ndarray]:
    # choose numeric feature columns (exclude date/ticker)
    feature_cols = [
        c
        for c in df.columns
        if c not in ["date", "ticker"] and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found; cannot fit PCA")

    # select and clean data
    X_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()

    # drop constant columns (zero variance)
    nunique = X_df.nunique(dropna=False)
    keep_cols = [c for c in feature_cols if nunique.get(c, 0) > 1]
    removed = [c for c in feature_cols if c not in keep_cols]
    if removed:
        # keep feature_cols consistent order after removal
        feature_cols = [c for c in feature_cols if c in keep_cols]
        X_df = X_df[feature_cols]

    # apply log1p to Volume if present to reduce scale/skew
    if "Volume" in X_df.columns:
        try:
            X_df["Volume"] = pd.to_numeric(X_df["Volume"], errors="coerce").fillna(0.0)
            X_df["Volume"] = np.log1p(X_df["Volume"])
        except Exception:
            pass

    # drop rows with NaNs after preprocessing
    X = X_df.dropna()
    if X.empty:
        raise ValueError("No numeric data available after cleaning; cannot fit PCA")

    # compute feature correlation (Pearson) on the cleaned inputs
    try:
        corr = X.corr().fillna(0.0)
        corr_matrix = corr.values.tolist()
    except Exception:
        corr_matrix = None

    # standardize features before PCA
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    n_comp = min(n_components, Xs.shape[1])
    pca = PCA(n_components=n_comp)
    comps = pca.fit_transform(Xs)

    try:
        dates = pd.to_datetime(df.loc[X.index, "date"]).dt.strftime("%Y-%m-%d").tolist()
    except Exception:
        dates = [str(i) for i in X.index.tolist()]

    meta = {
        "name": "pca_features",
        "created_at": datetime.now(UTC).isoformat(),
        "n_components": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_order": feature_cols,
        "row_index": dates,
        "feature_correlation": corr_matrix,
    }
    return meta, comps


def save_artifacts(
    meta: dict, comps: np.ndarray, registry_dir: str, ticker: str
) -> tuple[str, str]:
    os.makedirs(registry_dir, exist_ok=True)
    meta_path = os.path.join(registry_dir, f"pca_{ticker}.json")
    comps_path = os.path.join(registry_dir, f"pca_transformed_{ticker}.npy")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    np.save(comps_path, comps)
    return meta_path, comps_path


def main():
    parser = argparse.ArgumentParser(
        description="Train PCA embedding from feature data"
    )
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
    parser.add_argument(
        "--n-components", type=int, default=5, help="Number of PCA components"
    )
    args = parser.parse_args()

    df = load_features(args.features)
    ticker = os.path.splitext(os.path.basename(args.features))[0].upper()
    meta, comps = fit_pca(df, args.n_components)
    meta["ticker"] = ticker
    meta_path, comps_path = save_artifacts(meta, comps, args.registry, ticker)
    print(
        json.dumps(
            {"status": "ok", "pca_meta": meta_path, "pca_transformed": comps_path},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
