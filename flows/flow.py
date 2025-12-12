from prefect import flow

from .steps import (
    ingest,
    engineer,
    train_regressor,
    train_classification,
    compute_pca,
    cluster_features,
    forecast_ts,
)


@flow
def pipeline(ticker: str = "AAPL"):
    p = ingest(ticker)
    f = engineer(p)
    m = train_regressor(f)
    cls = train_classification(f)
    pca = compute_pca(f)
    clusters = cluster_features(f)
    fc = forecast_ts(f)
    return {
        "regression": m,
        "classification": cls,
        "pca": pca,
        "clusters": clusters,
        "forecast": fc,
    }


if __name__ == "__main__":
    pipeline()
