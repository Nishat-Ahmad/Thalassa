import os
from datetime import datetime
from prefect import flow

from .steps import (
    ingest,
    engineer,
    train_regressor,
    train_classification,
    train_association_rules,
    compute_pca,
    cluster_features,
    forecast_ts,
)

REGISTRY_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "registry")


@flow
def pipeline(ticker: str = "AAPL", run_dir: str | None = None):
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_dir = run_dir or os.path.join(REGISTRY_DIR, ticker.upper(), ts)
    os.makedirs(run_dir, exist_ok=True)
    p = ingest(ticker)
    f = engineer(p)
    m = train_regressor(f, run_dir=run_dir)
    cls = train_classification(f, run_dir=run_dir)
    assoc = train_association_rules(f, run_dir=run_dir)
    pca = compute_pca(f, run_dir=run_dir)
    clusters = cluster_features(f, run_dir=run_dir)
    fc = forecast_ts(f, run_dir=run_dir)
    return {
        "regression": m,
        "classification": cls,
        "association": assoc,
        "pca": pca,
        "clusters": clusters,
        "forecast": fc,
    }


if __name__ == "__main__":
    pipeline()
