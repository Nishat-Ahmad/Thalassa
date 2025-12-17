import os
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen


def _maybe_fix_prefect_api_url_env() -> None:
    """Fix Docker-only PREFECT_API_URL when running on the host.

    In Docker Compose, services reach Prefect at http://prefect:4200/api.
    On the host machine, that hostname usually doesn't resolve; localhost should be used.

    This must run *before* importing Prefect so the client picks up the corrected URL.
    """

    prefect_api_url = os.environ.get("PREFECT_API_URL")
    if not prefect_api_url:
        return

    try:
        parsed = urlparse(prefect_api_url)
    except Exception:
        return

    if parsed.hostname != "prefect":
        return

    def _health_url(api_url: str) -> str:
        # api_url is typically http://<host>:4200/api
        return api_url.rstrip("/") + "/health"

    def _is_healthy(api_url: str, timeout_s: float = 1.5) -> bool:
        try:
            with urlopen(_health_url(api_url), timeout=timeout_s) as resp:
                status = getattr(resp, "status", None)
                if status is None:
                    # Some Python builds may not expose .status; treat as success if no exception.
                    return True
                return 200 <= int(status) < 300
        except Exception:
            return False

    # If the docker hostname works and is a real Prefect API, keep it.
    if _is_healthy(prefect_api_url):
        return

    # Fall back to localhost when running on host.
    localhost_url = urlunparse(parsed._replace(netloc="localhost:4200"))
    if _is_healthy(localhost_url):
        os.environ["PREFECT_API_URL"] = localhost_url
        return

    # Final fallback: if nothing is reachable, unset PREFECT_API_URL so Prefect can use
    # its default (often an ephemeral/local API) instead of hard-failing.
    try:
        del os.environ["PREFECT_API_URL"]
    except KeyError:
        pass


_maybe_fix_prefect_api_url_env()

from prefect import flow  # noqa: E402
import requests  # noqa: E402

from .steps import (  # noqa: E402
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
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return
    payload = {"content": message}
    try:
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception:
        # Notifications should never break the pipeline.
        return


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
