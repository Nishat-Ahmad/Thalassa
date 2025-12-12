"""
Standalone ARIMA forecast trainer (mirrors forecast_ts in flows/flow.py).
Loads a feature parquet/CSV with a Close column, fits ARIMA(1,1,1), and writes
forecast.json into the registry directory.
"""

import argparse
import json
import os
from datetime import datetime, UTC

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception as e:  # pragma: no cover - missing dependency
    raise SystemExit("statsmodels is required for forecasting. Install statsmodels.") from e


def load_series(path: str) -> pd.Series:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "Close" not in df.columns:
        raise ValueError("Close column missing from features")
    return df.sort_values("date") if "date" in df.columns else df


def fit_arima(df: pd.DataFrame, horizon: int):
    series = df["Close"].astype(float)
    order = (1, 1, 1)
    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=horizon)
    conf_res = fitted.get_forecast(steps=horizon)
    conf = conf_res.conf_int()
    last_date = pd.to_datetime(df["date"].iloc[-1]) if "date" in df.columns else None
    idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D") if last_date is not None else list(range(1, horizon + 1))
    return fitted, forecast, conf, idx, order, series


def save_forecast(forecast, conf, idx, order, series, registry_dir: str):
    os.makedirs(registry_dir, exist_ok=True)
    out = {
        "name": "arima-forecast",
        "created_at": datetime.now(UTC).isoformat(),
        "horizon": int(len(forecast)),
        "order": list(order),
        "aic": float(getattr(forecast, 'aic', float('nan')))
        if hasattr(forecast, 'aic') else float('nan'),
        "bic": float(getattr(forecast, 'bic', float('nan')))
        if hasattr(forecast, 'bic') else float('nan'),
        "last_observation": float(series.iloc[-1]),
        "predictions": [float(x) for x in forecast.tolist()],
        "dates": [d.isoformat() if not isinstance(d, (int, float)) else str(d) for d in idx],
        "confidence_interval": conf.values.tolist(),
    }
    out_path = os.path.join(registry_dir, "forecast.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Train ARIMA forecast from feature data")
    parser.add_argument(
        "--features",
        default=os.path.join(os.path.dirname(__file__), "features", "AAPL.parquet"),
        help="Path to feature parquet/CSV file containing Close (and date) columns",
    )
    parser.add_argument(
        "--registry",
        default=os.path.join(os.path.dirname(__file__), "registry"),
        help="Output registry directory",
    )
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon in days")
    args = parser.parse_args()

    df = load_series(args.features)
    fitted, forecast, conf, idx, order, series = fit_arima(df, args.horizon)
    out_path = save_forecast(forecast, conf, idx, order, series, args.registry)
    print(json.dumps({"status": "ok", "forecast": out_path}, indent=2))


if __name__ == "__main__":
    main()
