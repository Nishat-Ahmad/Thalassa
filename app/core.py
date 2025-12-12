import os
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

MODEL_REGISTRY = os.path.join(BASE_DIR, "..", "ml", "registry")
XGB_META_PATH = os.path.join(MODEL_REGISTRY, "xgb_model.json")
XGB_MODEL_PATH = os.path.join(MODEL_REGISTRY, "xgb_model.ubj")
XGB_CLS_META_PATH = os.path.join(MODEL_REGISTRY, "xgb_classifier.json")
XGB_CLS_MODEL_PATH = os.path.join(MODEL_REGISTRY, "xgb_classifier.ubj")
PCA_META_PATH = os.path.join(MODEL_REGISTRY, "pca.json")
CLUSTERS_META_PATH = os.path.join(MODEL_REGISTRY, "clusters.json")
FORECAST_META_PATH = os.path.join(MODEL_REGISTRY, "forecast.json")

templates = Jinja2Templates(directory=TEMPLATES_DIR)