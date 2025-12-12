import os
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

MODEL_REGISTRY = os.path.join(BASE_DIR, "..", "ml", "registry")


def _safe_ticker(ticker: str | None) -> str:
	return (ticker or "AAPL").upper()


def xgb_paths(ticker: str | None = None) -> tuple[str, str]:
	t = _safe_ticker(ticker)
	model = os.path.join(MODEL_REGISTRY, f"xgb_model_{t}.ubj")
	meta = os.path.join(MODEL_REGISTRY, f"xgb_model_{t}.json")
	return model, meta


def xgb_classifier_paths(ticker: str | None = None) -> tuple[str, str]:
	t = _safe_ticker(ticker)
	model = os.path.join(MODEL_REGISTRY, f"xgb_classifier_{t}.ubj")
	meta = os.path.join(MODEL_REGISTRY, f"xgb_classifier_{t}.json")
	return model, meta


def pca_paths(ticker: str | None = None) -> tuple[str, str]:
	t = _safe_ticker(ticker)
	meta = os.path.join(MODEL_REGISTRY, f"pca_{t}.json")
	transformed = os.path.join(MODEL_REGISTRY, f"pca_transformed_{t}.npy")
	return meta, transformed


def cluster_paths(ticker: str | None = None) -> tuple[str, str]:
	t = _safe_ticker(ticker)
	meta = os.path.join(MODEL_REGISTRY, f"clusters_{t}.json")
	labels = os.path.join(MODEL_REGISTRY, f"cluster_labels_{t}.npy")
	return meta, labels


def forecast_path(ticker: str | None = None) -> str:
	t = _safe_ticker(ticker)
	return os.path.join(MODEL_REGISTRY, f"forecast_{t}.json")


def association_path(ticker: str | None = None) -> str:
	t = _safe_ticker(ticker)
	return os.path.join(MODEL_REGISTRY, f"association_{t}.json")

templates = Jinja2Templates(directory=TEMPLATES_DIR)