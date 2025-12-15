"""
Standalone association rules trainer (mirrors flows/association.py).
Loads a feature parquet/CSV, derives simple binarized signals, runs Apriori +
association_rules, and saves association_{ticker}.json to the registry directory.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def load_features(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    return df.sort_values("date") if "date" in df.columns else df


def build_itemset_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    items = {}
    if "Close" in cols:
        ret = df["Close"].pct_change().fillna(0.0)
        items["RET_UP"] = (ret > 0).astype(int)
        items["RET_DOWN"] = (ret < 0).astype(int)
        rolling = df["Close"].rolling(window=10, min_periods=5).mean()
        items["PRICE_ABOVE_SMA10"] = (df["Close"] > rolling).astype(int)
    if "Volume" in cols:
        vol_thr = df["Volume"].quantile(0.8)
        items["VOL_HIGH"] = (df["Volume"] >= vol_thr).astype(int)
    for c in cols:
        if df[c].dtype == "object":
            for val in df[c].dropna().unique()[:50]:
                items[f"{c}={val}"] = (df[c] == val).astype(int)
    item_df = pd.DataFrame(items, index=df.index)
    if item_df.empty:
        item_df["RET_UP"] = 0
    return item_df.clip(0, 1).astype(bool)


def train_association(
    df: pd.DataFrame, min_support: float, min_confidence: float, max_rules: int
):
    item_df = build_itemset_frame(df)
    freq = apriori(item_df, min_support=min_support, use_colnames=True)
    if freq.empty:
        return {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "rules": [],
            "n_rules": 0,
        }
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    # sort by confidence first (user preference), then lift and support
    rules = rules.sort_values(["confidence", "lift", "support"], ascending=False)
    # deduplicate symmetric rules (keep single direction per combined itemset)
    seen_itemsets = set()
    serialized = []
    for _, r in rules.iterrows():
        ant = (
            frozenset(r["antecedents"])
            if r.get("antecedents") is not None
            else frozenset()
        )
        cons = (
            frozenset(r["consequents"])
            if r.get("consequents") is not None
            else frozenset()
        )
        full = frozenset(list(ant) + list(cons))
        if not full or full in seen_itemsets:
            continue
        seen_itemsets.add(full)
        serialized.append(
            {
                "antecedents": sorted(list(ant)),
                "consequents": sorted(list(cons)),
                "support": float(r["support"]),
                "confidence": float(r.get("confidence", np.nan)),
                "lift": float(r.get("lift", np.nan)),
                "leverage": float(r.get("leverage", np.nan)),
                "conviction": float(r.get("conviction", np.nan)),
            }
        )
        if len(serialized) >= max_rules:
            break
    return {
        "min_support": min_support,
        "min_confidence": min_confidence,
        "n_rules": len(serialized),
        "rules": serialized,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train association rules from feature data"
    )
    parser.add_argument(
        "--features",
        default=os.path.join(os.path.dirname(__file__), "features", "AAPL.parquet"),
        help="Path to feature parquet/CSV file",
    )
    parser.add_argument(
        "--registry",
        default=os.path.join(os.path.dirname(__file__), "registry"),
        help="Output registry directory",
    )
    parser.add_argument(
        "--min-support", type=float, default=0.05, help="Minimum support for itemsets"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5, help="Minimum confidence for rules"
    )
    parser.add_argument("--max-rules", type=int, default=100, help="Max rules to keep")
    args = parser.parse_args()

    df = load_features(args.features)
    result = train_association(
        df, args.min_support, args.min_confidence, args.max_rules
    )
    ticker = os.path.splitext(os.path.basename(args.features))[0]
    out = {
        "ticker": ticker,
        **result,
    }

    os.makedirs(args.registry, exist_ok=True)
    out_path = os.path.join(args.registry, f"association_{ticker}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(
        json.dumps(
            {"status": "ok", "association": out_path, "n_rules": out["n_rules"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
