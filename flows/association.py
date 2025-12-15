import os
import json
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


REGISTRY_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "registry")
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "features")


def _output_path(ticker: str) -> str:
    return os.path.join(REGISTRY_DIR, f"association_{ticker.upper()}.json")


def _build_itemset_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Basic binarized signals derived from available numeric columns
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
    # Any categorical columns already present (encode each category as item)
    for c in cols:
        # skip ticker/symbol columns to avoid trivial ticker=... items
        if str(c).lower() in ("ticker", "symbol"):
            continue
        if df[c].dtype == "object":
            for val in df[c].dropna().unique()[:50]:
                items[f"{c}={val}"] = (df[c] == val).astype(int)
    item_df = pd.DataFrame(items, index=df.index)
    # ensure at least one column
    if item_df.empty:
        item_df["RET_UP"] = 0
    return item_df.clip(0, 1).astype(bool)


def compute_association_rules(
    ticker: str = "AAPL",
    min_support: float = 0.05,
    min_confidence: float = 0.5,
    max_rules: int = 200,
):
    feat_path = os.path.join(FEATURES_DIR, f"{ticker}.parquet")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Features file missing: {feat_path}")
    df = pd.read_parquet(feat_path).sort_values("date")
    item_df = _build_itemset_frame(df)
    # Apriori frequent itemsets
    freq = apriori(item_df, min_support=min_support, use_colnames=True)
    if freq.empty:
        result = {
            "ticker": ticker,
            "min_support": min_support,
            "min_confidence": min_confidence,
            "rules": [],
        }
    else:
        rules = association_rules(
            freq, metric="confidence", min_threshold=min_confidence
        )
        # prefer rules with higher confidence, then lift, then support
        rules = rules.sort_values(["confidence", "lift", "support"], ascending=False)
        # deduplicate symmetric rules coming from the same frequent itemset
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
        result = {
            "ticker": ticker,
            "min_support": min_support,
            "min_confidence": min_confidence,
            "n_rules": len(serialized),
            "rules": serialized,
        }
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(_output_path(ticker), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(compute_association_rules(), indent=2))
