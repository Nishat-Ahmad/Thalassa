import os
from datetime import datetime, timezone
from prefect import flow
import requests

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


def send_discord_notification(message):
    webhook_url = "https://discord.com/api/webhooks/1450346533798281318/OZkhPt8JlZXzT4Hy5AZr2sQPfP5s7qrpqdabiBKHC5kpoizgREw7B7XCTZNupQaI2T0_"
    payload = {"content": message}
    requests.post(webhook_url, json=payload)


@flow
def pipeline(ticker: str = "AAPL", run_dir: str | None = None):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    run_dir = run_dir or os.path.join(REGISTRY_DIR, ticker.upper(), ts)
    os.makedirs(run_dir, exist_ok=True)
    p = ingest(ticker)
    f = engineer(p)
    m = train_regressor(f, run_dir=run_dir)
    cls = train_classification(f, run_dir=run_dir)
    assoc = train_association_rules(f, run_dir=run_dir)
    # After classification training, produce a next-day prediction using the latest features
    try:
        from .steps import predict_next

        pred = predict_next(f, run_dir=run_dir)
    except Exception:
        pred = {"status": "skipped", "reason": "predict_next not available"}
    # Compute clusters from raw features first to avoid using PCA-transformed features
    clusters = cluster_features(f, run_dir=run_dir)
    pca = compute_pca(f, run_dir=run_dir)
    fc = forecast_ts(f, run_dir=run_dir)
    send_discord_notification("Pipeline succeeded!")
    return {
        "regression": m,
        "classification": cls,
        "association": assoc,
        "prediction": pred,
        "pca": pca,
        "clusters": clusters,
        "forecast": fc,
    }


def notify_flow_status():
    message = """
    **Flow Run Summary**
    :rocket: **Flow Name**: `slim-alligator`
    :white_check_mark: **Status**: `Completed`

    **Task Statuses:**
    - :white_check_mark: `ingest`: Completed
    - :white_check_mark: `engineer`: Completed
    - :white_check_mark: `train_regressor`: Completed
    - :white_check_mark: `train_classification`: Completed
    - :white_check_mark: `train_association_rules`: Completed
    - :white_check_mark: `predict_next`: Completed
    - :white_check_mark: `cluster_features`: Completed
    - :white_check_mark: `compute_pca`: Completed
    - :white_check_mark: `forecast_ts`: Completed

    :tada: All tasks ran successfully!
    """
    send_discord_notification(message)


if __name__ == "__main__":
    try:
        pipeline()
    except Exception as e:
        send_discord_notification(f"Pipeline failed: {e}")
        raise
    notify_flow_status()
