import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _utc_ts() -> str:
	return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _safe_mkdir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def _read_parquet(path: str) -> pd.DataFrame:
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	if os.path.getsize(path) == 0:
		raise ValueError(f"Parquet file is 0 bytes: {path}")
	return pd.read_parquet(path)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
	if isinstance(df.columns, pd.MultiIndex):
		df = df.copy()
		df.columns = [" ".join([str(x) for x in tup if str(x) != ""]) for tup in df.columns]
	return df


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
	numeric = df.select_dtypes(include=[np.number]).copy()
	numeric = numeric.replace([np.inf, -np.inf], np.nan)
	return numeric


def _approx_ks_pvalue(d: float, n1: int, n2: int) -> float:
	"""Asymptotic approximation of KS two-sample p-value.

	This avoids a scipy dependency. It's good enough for simple gating.
	"""
	if n1 <= 0 or n2 <= 0:
		return 1.0
	en = np.sqrt(n1 * n2 / (n1 + n2))
	# Kolmogorov distribution approximation
	lam = (en + 0.12 + 0.11 / en) * d
	if lam <= 0:
		return 1.0
	# p ≈ 2 * sum_{j=1..inf} (-1)^{j-1} exp(-2 j^2 lam^2)
	s = 0.0
	for j in range(1, 101):
		term = 2.0 * ((-1.0) ** (j - 1)) * np.exp(-2.0 * (j * j) * (lam * lam))
		s += term
		if abs(term) < 1e-10:
			break
	# clamp
	return float(min(max(s, 0.0), 1.0))


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
	"""Two-sample KS statistic D (no scipy)."""
	x = x[~np.isnan(x)]
	y = y[~np.isnan(y)]
	if len(x) == 0 or len(y) == 0:
		return 0.0
	x = np.sort(x)
	y = np.sort(y)
	data_all = np.sort(np.concatenate([x, y]))
	cdf_x = np.searchsorted(x, data_all, side="right") / len(x)
	cdf_y = np.searchsorted(y, data_all, side="right") / len(y)
	return float(np.max(np.abs(cdf_x - cdf_y)))


@dataclass
class CheckResult:
	name: str
	severity: str  # "ok" | "warning" | "severe"
	details: Dict[str, Any]


def _integrity_checks(df: pd.DataFrame) -> List[CheckResult]:
	results: List[CheckResult] = []
	results.append(
		CheckResult(
			name="file_non_empty",
			severity="ok" if len(df) > 0 else "severe",
			details={"rows": int(len(df)), "cols": int(df.shape[1])},
		)
	)

	# Missingness
	numeric = _numeric_frame(df)
	missing_rate = float(numeric.isna().mean().mean()) if numeric.shape[1] else 0.0
	results.append(
		CheckResult(
			name="missingness",
			severity="severe" if missing_rate > 0.30 else ("warning" if missing_rate > 0.10 else "ok"),
			details={"mean_missing_rate": missing_rate},
		)
	)

	# Duplicates
	dup_rows = int(df.duplicated().sum())
	results.append(
		CheckResult(
			name="duplicate_rows",
			severity="warning" if dup_rows > 0 else "ok",
			details={"duplicates": dup_rows},
		)
	)

	# Date monotonicity (best-effort)
	date_col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
	if date_col:
		try:
			dates = pd.to_datetime(df[date_col], errors="coerce")
			non_null = dates.dropna()
			is_monotonic = bool(non_null.is_monotonic_increasing)
			sev = "ok" if is_monotonic else "warning"
			results.append(
				CheckResult(
					name="date_monotonic",
					severity=sev,
					details={"date_col": date_col, "monotonic_increasing": is_monotonic},
				)
			)
		except Exception as e:
			results.append(
				CheckResult(
					name="date_monotonic",
					severity="warning",
					details={"date_col": date_col, "error": str(e)},
				)
			)

	return results


def _drift_checks(
	baseline: pd.DataFrame,
	current: pd.DataFrame,
	metric: str,
	pvalue_threshold: float,
	window: Optional[int] = None,
) -> List[CheckResult]:
	if metric.lower() != "ks":
		return [
			CheckResult(
				name="drift_metric",
				severity="warning",
				details={"metric": metric, "note": "Only 'ks' is supported; drift checks skipped."},
			)
		]

	base = _numeric_frame(_flatten_columns(baseline))
	cur = _numeric_frame(_flatten_columns(current))

	# Optional window: last N rows (assumes time-sorted)
	if window and window > 0:
		cur = cur.tail(int(window))

	shared_cols = [c for c in cur.columns if c in base.columns]
	if not shared_cols:
		return [
			CheckResult(
				name="drift_shared_columns",
				severity="warning",
				details={"shared_numeric_columns": 0},
			)
		]

	per_col: Dict[str, Dict[str, float]] = {}
	severe_cols: List[str] = []
	for col in shared_cols:
		x = base[col].to_numpy(dtype=float, copy=False)
		y = cur[col].to_numpy(dtype=float, copy=False)
		d = _ks_statistic(x, y)
		p = _approx_ks_pvalue(d, int(np.sum(~np.isnan(x))), int(np.sum(~np.isnan(y))))
		per_col[col] = {"ks_d": float(d), "p_value": float(p)}
		# heuristic thresholds
		if (p < pvalue_threshold) and (d > 0.15):
			severe_cols.append(col)

	frac_severe = len(severe_cols) / max(len(shared_cols), 1)
	severity = "severe" if frac_severe >= 0.25 else ("warning" if frac_severe > 0.0 else "ok")

	return [
		CheckResult(
			name="data_drift",
			severity=severity,
			details={
				"metric": "ks",
				"pvalue_threshold": float(pvalue_threshold),
				"shared_columns": int(len(shared_cols)),
				"severe_columns": severe_cols,
				"fraction_severe": float(frac_severe),
				"per_column": per_col,
			},
		)
	]


def _find_latest_run_dir(registry_root: str, ticker: str) -> Optional[str]:
	ticker_dir = os.path.join(registry_root, ticker.upper())
	if not os.path.isdir(ticker_dir):
		return None
	candidates = [
		os.path.join(ticker_dir, d)
		for d in os.listdir(ticker_dir)
		if os.path.isdir(os.path.join(ticker_dir, d))
	]
	if not candidates:
		return ticker_dir
	return sorted(candidates)[-1]


def _load_xgb_regressor(run_dir: str, ticker: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
	meta_path = os.path.join(run_dir, f"xgb_model_{ticker.upper()}.json")
	ubj_path = os.path.join(run_dir, f"xgb_model_{ticker.upper()}.ubj")
	if not (os.path.exists(meta_path) and os.path.exists(ubj_path)):
		return None, None
	with open(meta_path, "r") as f:
		meta = json.load(f)
	try:
		import xgboost as xgb

		booster = xgb.Booster()
		booster.load_model(ubj_path)
		return booster, meta
	except Exception:
		return None, meta


def _load_xgb_classifier(run_dir: str, ticker: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
	meta_path = os.path.join(run_dir, f"xgb_classifier_{ticker.upper()}.json")
	ubj_path = os.path.join(run_dir, f"xgb_classifier_{ticker.upper()}.ubj")
	if not (os.path.exists(meta_path) and os.path.exists(ubj_path)):
		return None, None
	with open(meta_path, "r") as f:
		meta = json.load(f)
	try:
		import xgboost as xgb

		booster = xgb.Booster()
		booster.load_model(ubj_path)
		return booster, meta
	except Exception:
		return None, meta


def _performance_checks(
	df: pd.DataFrame,
	registry_root: str,
	ticker: str,
) -> List[CheckResult]:
	results: List[CheckResult] = []
	df = _flatten_columns(df)
	num = _numeric_frame(df)
	# Regression: predict next-day return
	run_dir = _find_latest_run_dir(registry_root, ticker)
	if not run_dir:
		results.append(
			CheckResult(
				name="model_artifacts",
				severity="warning",
				details={"note": "No registry run directory found; performance checks skipped."},
			)
		)
		return results

	reg_model, reg_meta = _load_xgb_regressor(run_dir, ticker)
	if reg_meta is None:
		results.append(
			CheckResult(
				name="regressor_artifacts",
				severity="warning",
				details={"note": "Regressor artifacts not found; skipping regressor checks.", "run_dir": run_dir},
			)
		)
	else:
		# Build X and y using the same definition as training
		if "return" in num.columns:
			y = num["return"].shift(-1)
		else:
			y = None

		feature_cols = [c for c in reg_meta.get("features", []) if c in num.columns]
		if not feature_cols:
			feature_cols = [c for c in num.columns if c not in ["return"]]

		X = num[feature_cols]
		if y is not None:
			mask = (~X.isna().any(axis=1)) & (~y.isna())
			X = X.loc[mask]
			y = y.loc[mask]
		else:
			mask = ~X.isna().any(axis=1)
			X = X.loc[mask]

		# Require enough rows to evaluate
		if len(X) < 50 or y is None or len(y) < 50:
			results.append(
				CheckResult(
					name="regressor_performance",
					severity="warning",
					details={"note": "Not enough labeled samples for regression eval.", "samples": int(len(X))},
				)
			)
		elif reg_model is None:
			results.append(
				CheckResult(
					name="regressor_performance",
					severity="warning",
					details={"note": "xgboost not available in environment; cannot score regressor."},
				)
			)
		else:
			import xgboost as xgb

			try:
				expected = int(reg_model.num_features())
			except Exception:
				expected = None
			if expected is not None and expected != int(X.shape[1]):
				results.append(
					CheckResult(
						name="regressor_feature_mismatch",
						severity="warning",
						details={"expected_features": expected, "provided_features": int(X.shape[1])},
					)
				)
			else:
				split = int(0.8 * len(X))
				X_test = X.iloc[split:]
				y_test = y.iloc[split:]
				dtest = xgb.DMatrix(X_test, feature_names=[str(c) for c in X_test.columns])
				pred = reg_model.predict(dtest)
				rmse = float(np.sqrt(np.mean((y_test.to_numpy() - pred) ** 2)))
				mae = float(np.mean(np.abs(y_test.to_numpy() - pred)))
				baseline_rmse = None
				try:
					baseline_rmse = float(reg_meta.get("metrics", {}).get("rmse"))
				except Exception:
					baseline_rmse = None
				# Gate: compare to training metric if present
				severe = False
				if baseline_rmse is not None and baseline_rmse > 0:
					severe = rmse > (baseline_rmse * 3.0)
				else:
					severe = rmse > 0.10
				results.append(
					CheckResult(
						name="regressor_performance",
						severity="severe" if severe else "ok",
						details={"rmse": rmse, "mae": mae, "baseline_rmse": baseline_rmse},
					)
				)

	# Classification
	cls_model, cls_meta = _load_xgb_classifier(run_dir, ticker)
	if cls_meta is None:
		results.append(
			CheckResult(
				name="classifier_artifacts",
				severity="warning",
				details={"note": "Classifier artifacts not found; skipping classifier checks.", "run_dir": run_dir},
			)
		)
	else:
		# Labels: up/down based on log_return if present, else return.
		if "log_return" in num.columns:
			y_cls = (num["log_return"] > 0).astype(int)
		elif "return" in num.columns:
			y_cls = (num["return"] > 0).astype(int)
		else:
			y_cls = None

		feature_cols = [c for c in cls_meta.get("features", []) if c in num.columns]
		if not feature_cols:
			feature_cols = [c for c in num.columns if c not in ["return", "log_return"]]
		Xc = num[feature_cols]
		if y_cls is None:
			results.append(
				CheckResult(
					name="classifier_performance",
					severity="warning",
					details={"note": "No target column for classification eval (need log_return or return)."},
				)
			)
		else:
			mask = (~Xc.isna().any(axis=1)) & (~y_cls.isna())
			Xc = Xc.loc[mask]
			y_cls = y_cls.loc[mask]
			if len(Xc) < 50:
				results.append(
					CheckResult(
						name="classifier_performance",
						severity="warning",
						details={"note": "Not enough samples for classification eval.", "samples": int(len(Xc))},
					)
				)
			elif cls_model is None:
				results.append(
					CheckResult(
						name="classifier_performance",
						severity="warning",
						details={"note": "xgboost not available in environment; cannot score classifier."},
					)
				)
			else:
				import xgboost as xgb

				try:
					expected = int(cls_model.num_features())
				except Exception:
					expected = None
				if expected is not None and expected != int(Xc.shape[1]):
					results.append(
						CheckResult(
							name="classifier_feature_mismatch",
							severity="warning",
							details={"expected_features": expected, "provided_features": int(Xc.shape[1])},
						)
					)
				else:
					split = int(0.8 * len(Xc))
					X_test = Xc.iloc[split:]
					y_test = y_cls.iloc[split:]
					dtest = xgb.DMatrix(X_test, feature_names=[str(c) for c in X_test.columns])
					pred = cls_model.predict(dtest)
					eps = 1e-15
					logloss = float(
						np.mean(
							-(
								y_test.to_numpy() * np.log(pred + eps)
								+ (1 - y_test.to_numpy()) * np.log(1 - pred + eps)
							)
						)
					)
					auc = None
					try:
						from sklearn.metrics import roc_auc_score

						auc = float(roc_auc_score(y_test.to_numpy(), pred))
					except Exception:
						auc = None
					severe = logloss > 1.0
					if auc is not None:
						severe = severe or (auc < 0.50)
					results.append(
						CheckResult(
							name="classifier_performance",
							severity="severe" if severe else "ok",
							details={"logloss": logloss, "auc": auc},
						)
					)

	return results


def _render_markdown(report: Dict[str, Any]) -> str:
	lines: List[str] = []
	lines.append(f"# DeepChecks Report ({report['ticker']})")
	lines.append("")
	lines.append(f"- Timestamp (UTC): {report['timestamp_utc']}")
	lines.append(f"- Features: {report['features_path']}")
	lines.append(f"- Registry: {report['registry_root']}")
	lines.append(f"- Overall status: **{report['overall_status']}**")
	lines.append("")
	lines.append("## Checks")
	for chk in report["checks"]:
		lines.append(f"- **{chk['name']}**: `{chk['severity']}`")
		# keep details compact
		details = chk.get("details", {})
		if details:
			short = json.dumps(details, ensure_ascii=False)[:800]
			lines.append(f"  - details: {short}")
	lines.append("")
	return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="DeepChecks runner for Thalassa")
	parser.add_argument("--features", required=True, help="Path to features parquet")
	parser.add_argument("--registry", required=True, help="Path to ml/registry")
	parser.add_argument("--ticker", required=True, help="Ticker symbol")
	parser.add_argument("--fail-on-severe", action="store_true", help="Exit non-zero if severe issues found")
	parser.add_argument("--feature-mode", default="engineered", choices=["engineered", "raw"], help="Used for reporting; affects drift windowing")
	parser.add_argument("--window", type=int, default=0, help="If set, evaluate drift on last N rows")
	parser.add_argument("--metric", default="ks", help="Drift metric (currently supports: ks)")
	parser.add_argument("--pvalue-threshold", type=float, default=0.001, help="p-value threshold for drift")
	args = parser.parse_args(argv)

	report_dir = os.path.join(args.registry, "deepchecks")
	_safe_mkdir(report_dir)
	baseline_dir = os.path.join(report_dir, "baseline")
	_safe_mkdir(baseline_dir)
	baseline_path = os.path.join(baseline_dir, f"{args.ticker.upper()}.parquet")

	ts = _utc_ts()
	try:
		df = _flatten_columns(_read_parquet(args.features))
	except Exception as e:
		report = {
			"timestamp_utc": ts,
			"ticker": args.ticker.upper(),
			"features_path": args.features,
			"registry_root": args.registry,
			"overall_status": "severe",
			"checks": [
				{"name": "load_features", "severity": "severe", "details": {"error": str(e)}}
			],
		}
		out_json = os.path.join(report_dir, f"report_{args.ticker.upper()}_{ts}.json")
		with open(out_json, "w", encoding="utf-8") as f:
			json.dump(report, f, indent=2)
		out_md = os.path.join(report_dir, f"report_{args.ticker.upper()}_{ts}.md")
		with open(out_md, "w", encoding="utf-8") as f:
			f.write(_render_markdown(report))
		return 2 if args.fail_on_severe else 0

	checks: List[CheckResult] = []
	checks.extend(_integrity_checks(df))
	checks.extend(_performance_checks(df, args.registry, args.ticker))

	# Drift: compare to baseline if present; if not, create baseline and don't fail
	if os.path.exists(baseline_path) and os.path.getsize(baseline_path) > 0:
		try:
			baseline_df = _flatten_columns(_read_parquet(baseline_path))
			checks.extend(
				_drift_checks(
					baseline=baseline_df,
					current=df,
					metric=args.metric,
					pvalue_threshold=args.pvalue_threshold,
					window=args.window if args.window > 0 else None,
				)
			)
		except Exception as e:
			checks.append(
				CheckResult(
					name="drift_load_baseline",
					severity="warning",
					details={"baseline_path": baseline_path, "error": str(e)},
				)
			)
	else:
		# Initialize baseline
		try:
			df.to_parquet(baseline_path)
			checks.append(
				CheckResult(
					name="baseline_initialized",
					severity="ok",
					details={"baseline_path": baseline_path},
				)
			)
		except Exception as e:
			checks.append(
				CheckResult(
					name="baseline_initialized",
					severity="warning",
					details={"baseline_path": baseline_path, "error": str(e)},
				)
			)

	overall_status = "ok"
	if any(c.severity == "severe" for c in checks):
		overall_status = "severe"
	elif any(c.severity == "warning" for c in checks):
		overall_status = "warning"

	report = {
		"timestamp_utc": ts,
		"ticker": args.ticker.upper(),
		"features_path": args.features,
		"registry_root": args.registry,
		"feature_mode": args.feature_mode,
		"overall_status": overall_status,
		"checks": [
			{"name": c.name, "severity": c.severity, "details": c.details} for c in checks
		],
	}

	out_json = os.path.join(report_dir, f"report_{args.ticker.upper()}_{ts}.json")
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)
	out_md = os.path.join(report_dir, f"report_{args.ticker.upper()}_{ts}.md")
	with open(out_md, "w", encoding="utf-8") as f:
		f.write(_render_markdown(report))

	if args.fail_on_severe and overall_status == "severe":
		return 2
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
