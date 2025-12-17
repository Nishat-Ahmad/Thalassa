# Finance ML MLOps Project (Thalassa)

Domain: Economics & Finance

End-to-end ML Engineering system that combines:
- FastAPI web app + API endpoints (JSON + file upload)
- Prefect orchestration (ETL, feature engineering, multi-task ML)
- Automated ML checks (data integrity, drift, performance)
- CI/CD with GitHub Actions (tests → pipeline → checks → Docker build)
- Docker + Docker Compose (API + Prefect Server + worker)

## What this project does

### ML tasks included in the workflow
The Prefect flow runs multiple ML tasks in one pipeline:
- Regression (XGBoost regressor)
- Classification (XGBoost classifier)
- Dimensionality reduction (PCA)
- Clustering (KMeans)
- Time series forecasting (ARIMA)
- Association rule mining
- Lightweight recommendation (nearest neighbors in PCA space)

### Artifacts & versioning
Artifacts are saved under `ml/registry/<TICKER>/<UTC_TIMESTAMP>/` (plus some backward-compatible flat files).

Key data paths:
- `ml/data/` – downloaded raw price data (Parquet)
- `ml/features/` – engineered features (Parquet)
- `ml/registry/` – trained models + metadata + PCA/clusters/forecast outputs
- `ml/registry/deepchecks/` – ML check reports (JSON/Markdown)

## Quickstart (Local, no Docker)

Prereqs:
- Python 3.11

Install dependencies:
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the API:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run the Prefect pipeline once (creates/updates artifacts):
```bash
python tools/run_pipeline.py
```

## Quickstart (Docker Compose: API + Prefect)

This Compose setup runs:
- Prefect server (UI/API) at `http://localhost:4200`
- FastAPI at `http://localhost:8000`
- Prefect worker (process work pool)
- One-shot bootstrap that creates a work pool + registers a scheduled deployment

Start everything:
```bash
docker compose up --build -d
```

Verify services:
```bash
curl http://localhost:8000/health
curl http://localhost:4200/api/health
```

Stop:
```bash
docker compose down
```

Notes:
- `ml/data`, `ml/features`, `ml/registry` are stored in named Docker volumes so the API and worker share artifacts.
- The `prefect-init` service is expected to exit after successful bootstrap.

## API Overview

The FastAPI app exposes endpoints for:
- Health/status: `GET /health`
- Trigger pipeline: `POST /run-pipeline?ticker=AAPL`
- Regression prediction (JSON): `POST /predict`
- Classification prediction (JSON): `POST /predict-class`
- File upload: `POST /upload-csv`
- Batch prediction: `POST /predict-batch`
- Forecast results: `GET /forecast?ticker=AAPL`
- PCA/cluster metadata: `GET /pca-info`, `GET /cluster-info`
- Cluster utilities: `GET /cluster-samples`, `POST /predict-cluster`
- Recommendation: `GET /recommend`, `POST /recommend`
- Association rules: `GET /association-info`

OpenAPI docs:
- Swagger UI: `http://localhost:8000/docs`

## Prefect orchestration

Main flow entry point:
- `flows/flow.py:pipeline`

Pipeline steps (high-level):
1. Download data (yfinance)
2. Engineer features
3. Train regressor + classifier
4. Generate association rules
5. Cluster features + PCA
6. Forecast time series

Retries:
- Ingestion task has Prefect retries and internal retries.

Scheduling:
- Compose registers a scheduled deployment via `tools/prefect_bootstrap.py`.

## Automated ML checks (DeepChecks-equivalent)

This repo includes an internal “DeepChecks runner” that produces a report and can gate CI.

It implements:
- Data integrity checks (non-empty, missingness, duplicates, date monotonicity)
- Drift checks (KS-test style heuristic vs baseline)
- Performance checks (evaluate model vs stored baseline metrics when available)

Run locally:
```bash
python ml/deepchecks/run_deepchecks.py --features ml/features/AAPL.parquet --registry ml/registry --ticker AAPL --fail-on-severe
```

Baseline behavior:
- First run creates `ml/registry/deepchecks/baseline/<TICKER>.parquet` if missing.

## Testing

Run tests:
```bash
pytest -q
```

Notes:
- Tests are designed to be CI-friendly and avoid requiring network access.

## CI/CD (GitHub Actions)

Workflow file:
- `.github/workflows/ci.yml`

High-level jobs:
- Lint + formatting checks
- Unit/ML tests
- Run Prefect pipeline (produces artifacts)
- Run ML checks and upload reports
- Build & push Docker image (GHCR)

## Configuration (Environment Variables)

Common:
- `PREFECT_API_URL` – Prefect API (Compose defaults to `http://prefect:4200/api` in containers)
- `PREFECT_WORK_POOL` – work pool name (default: `thalassa-pool`)

## Troubleshooting

### Parquet errors ("Unable to find a usable engine")
Pandas requires a Parquet engine.

This project uses `pyarrow` and it is included in `requirements.txt`. If you rebuild the image after pulling changes, Parquet write/read should work.

Sanity check inside Docker:
```bash
docker run --rm finance:latest python -c "import pyarrow, pandas as pd; print('pyarrow', pyarrow.__version__); print('pandas', pd.__version__)"
```

### Network/data-source failures (yfinance)
Ticker downloads can fail due to rate limiting or network issues. The ingest task retries and uses timeouts.

## Security note

Do NOT commit secrets (API keys/webhooks) into source control. Use environment variables or secret managers for any notification integrations.
