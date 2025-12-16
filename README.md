# Finance ML MLOps Project

Domain: Economics & Finance

This repository will evolve into a full end-to-end ML engineering platform featuring:
- FastAPI service for prediction endpoints
- Prefect flows for ingestion, feature engineering, training, evaluation
- Model registry & versioning
- CI/CD via GitHub Actions
- Docker & optional Compose orchestration
- Automated ML quality gates (data integrity, drift, performance)

## Current Status
Website skeleton + baseline API + initial Prefect pipeline (ingestion, indicators, baseline training).

## Running
```bash
uvicorn app.main:app --reload --app-dir d:\Code\Finance
python flows/flow.py  # run ETL + baseline train
```

## Docker Compose (Prefect + API, end-to-end)

This repo includes a minimal-but-complete Compose setup that runs:
- Prefect server (UI/API) on `http://localhost:4200`
- API on `http://localhost:8000`
- A Prefect worker bound to a work pool (`thalassa-pool`)
- A one-shot bootstrap container that registers a scheduled deployment

```bash
docker compose up --build
```

To (re)register the scheduled deployment explicitly:

```bash
docker compose run --rm prefect-init
```

Artifacts (`ml/registry`, `ml/features`, `ml/data`) are stored in named volumes so the API and worker share the same state.

## Next Steps
1. Expand feature engineering & persist metadata
2. Add XGBoost models (regression/classification)
3. Introduce PCA, clustering, recommendation modules
4. Integrate Deepchecks in CI
5. Add monitoring & drift detection scheduled via Prefect
