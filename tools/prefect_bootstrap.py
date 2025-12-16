import os
import subprocess
import sys
import time
from typing import List, Optional

import requests


def _run(cmd: List[str], *, allow_fail: bool = False) -> int:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(proc.returncode)
    return proc.returncode


def _wait_for_prefect(api_url: str, timeout_s: int = 120) -> None:
    # Prefect server exposes health at /api/health in most 2.x versions.
    # Some versions may behave slightly differently, so accept any 2xx.
    health_url = api_url.rstrip("/") + "/health"
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None

    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=5)
            if 200 <= r.status_code < 300:
                print(f"Prefect API healthy: {health_url}", flush=True)
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2)

    raise RuntimeError(f"Prefect API not healthy after {timeout_s}s: {health_url} (last: {last_err})")


def _ensure_work_pool(pool: str) -> None:
    # Prefect 3 work pools are upsert-able with --overwrite.
    _run(
        [
            "prefect",
            "--no-prompt",
            "work-pool",
            "create",
            pool,
            "--type",
            "process",
            "--overwrite",
        ],
        allow_fail=False,
    )


def _apply_deployment(pool: str, name: str, cron: str, tz: str) -> None:
    # Prefect 3 uses `prefect deploy` (and a `prefect.yaml` is optional).
    # Entry point: flows/flow.py:pipeline
    base = [
        "prefect",
        "--no-prompt",
        "deploy",
        "flows/flow.py:pipeline",
        "-n",
        name,
        "-p",
        pool,
        "--cron",
        cron,
        "--timezone",
        tz,
    ]
    _run(base)


def main() -> int:
    api_url = os.environ.get("PREFECT_API_URL", "http://prefect:4200/api").rstrip("/")
    pool = os.environ.get("PREFECT_WORK_POOL", "thalassa-pool")
    deployment = os.environ.get("PREFECT_DEPLOYMENT_NAME", "scheduled")
    cron = os.environ.get("PREFECT_SCHEDULE_CRON", "0 * * * *")  # hourly
    tz = os.environ.get("PREFECT_SCHEDULE_TZ", "UTC")

    # Ensure the CLI talks to the right API.
    os.environ["PREFECT_API_URL"] = api_url

    print(f"Using PREFECT_API_URL={api_url}", flush=True)
    _wait_for_prefect(api_url)

    _ensure_work_pool(pool)
    _apply_deployment(pool=pool, name=deployment, cron=cron, tz=tz)

    print("Prefect bootstrap complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
