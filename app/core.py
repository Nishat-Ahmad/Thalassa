import os
from datetime import datetime
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

MODEL_REGISTRY = os.path.join(BASE_DIR, "..", "ml", "registry")


def _safe_ticker(ticker: str | None) -> str:
	return (ticker or "AAPL").upper()


def ensure_run_dir(ticker: str, timestamp: str | None = None) -> str:
	t = _safe_ticker(ticker)
	base = os.path.join(MODEL_REGISTRY, t)
	os.makedirs(base, exist_ok=True)
	ts = timestamp or datetime.utcnow().strftime("%Y%m%d%H%M%S")
	run_dir = os.path.join(base, ts)
	os.makedirs(run_dir, exist_ok=True)
	return run_dir


def _resolve_run_dir(ticker: str, run_dir: str | None = None) -> str:
	"""
	Return the target run directory for a ticker. If run_dir is provided, use it; otherwise
	pick the latest timestamped run directory under MODEL_REGISTRY/<ticker>. Fallback to
	the flat MODEL_REGISTRY for backward compatibility when no run directories exist.
	"""
	t = _safe_ticker(ticker)
	if run_dir:
		return run_dir
	base = os.path.join(MODEL_REGISTRY, t)
	if os.path.isdir(base):
		candidates = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
		if candidates:
			latest = sorted(candidates)[-1]
			return os.path.join(base, latest)
	return MODEL_REGISTRY


def xgb_paths(ticker: str | None = None, run_dir: str | None = None) -> tuple[str, str]:
	t = _safe_ticker(ticker)
	rd = _resolve_run_dir(t, run_dir)
	model = os.path.join(rd, f"xgb_model_{t}.ubj")
	meta = os.path.join(rd, f"xgb_model_{t}.json")
	if run_dir is None and not os.path.exists(model):
		# backward compatibility for flat registry
		model = os.path.join(MODEL_REGISTRY, f"xgb_model_{t}.ubj")
		meta = os.path.join(MODEL_REGISTRY, f"xgb_model_{t}.json")
	return model, meta


def xgb_classifier_paths(ticker: str | None = None, run_dir: str | None = None) -> tuple[str, str]:
	t = _safe_ticker(ticker)
	rd = _resolve_run_dir(t, run_dir)
	model = os.path.join(rd, f"xgb_classifier_{t}.ubj")
	meta = os.path.join(rd, f"xgb_classifier_{t}.json")
	if run_dir is None and not os.path.exists(model):
		model = os.path.join(MODEL_REGISTRY, f"xgb_classifier_{t}.ubj")
		meta = os.path.join(MODEL_REGISTRY, f"xgb_classifier_{t}.json")
	return model, meta


def pca_paths(ticker: str | None = None, run_dir: str | None = None) -> tuple[str, str]:
	t = _safe_ticker(ticker)
	rd = _resolve_run_dir(t, run_dir)
	# Prefer the resolved run dir first
	meta = os.path.join(rd, f"pca_{t}.json")
	transformed = os.path.join(rd, f"pca_transformed_{t}.npy")

	# If a specific run_dir was not provided and the expected meta is missing,
	# search timestamped subdirectories under MODEL_REGISTRY/<ticker> for the
	# newest PCA metadata and use that. This handles cases where artifacts are
	# created after the app started or when run-dir resolution didn't pick the
	# correct subfolder.
	if run_dir is None and not os.path.exists(meta):
		base = os.path.join(MODEL_REGISTRY, t)
		try:
			if os.path.isdir(base):
				candidates = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
				# find any candidate that contains pca metadata
				pca_candidates = []
				for c in candidates:
					m = os.path.join(c, f"pca_{t}.json")
					tr = os.path.join(c, f"pca_transformed_{t}.npy")
					if os.path.exists(m):
						pca_candidates.append((c, m, tr))
				if pca_candidates:
					# choose the candidate with the lexicographically largest folder name (timestamp)
					best = sorted(pca_candidates, key=lambda x: x[0])[-1]
					meta = best[1]
					transformed = best[2]
					return meta, transformed
		except Exception:
			pass

	# fallback to flat registry for backward compatibility
	if run_dir is None and not os.path.exists(meta):
		meta = os.path.join(MODEL_REGISTRY, f"pca_{t}.json")
		transformed = os.path.join(MODEL_REGISTRY, f"pca_transformed_{t}.npy")
	return meta, transformed


def cluster_paths(ticker: str | None = None, run_dir: str | None = None) -> tuple[str, str]:
	t = _safe_ticker(ticker)
	rd = _resolve_run_dir(t, run_dir)
	meta = os.path.join(rd, f"clusters_{t}.json")
	labels = os.path.join(rd, f"cluster_labels_{t}.npy")
	if run_dir is None and not os.path.exists(meta):
		meta = os.path.join(MODEL_REGISTRY, f"clusters_{t}.json")
		labels = os.path.join(MODEL_REGISTRY, f"cluster_labels_{t}.npy")
	return meta, labels


def forecast_path(ticker: str | None = None, run_dir: str | None = None) -> str:
	t = _safe_ticker(ticker)
	rd = _resolve_run_dir(t, run_dir)
	path = os.path.join(rd, f"forecast_{t}.json")
	if run_dir is None and not os.path.exists(path):
		path = os.path.join(MODEL_REGISTRY, f"forecast_{t}.json")
	return path


def association_path(ticker: str | None = None, run_dir: str | None = None) -> str:
	t = _safe_ticker(ticker)
	rd = _resolve_run_dir(t, run_dir)
	path = os.path.join(rd, f"association_{t}.json")
	if run_dir is None and not os.path.exists(path):
		path = os.path.join(MODEL_REGISTRY, f"association_{t}.json")
	return path

templates = Jinja2Templates(directory=TEMPLATES_DIR)