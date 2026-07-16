"""
Segment x Risk x Value matrix.

This is the interpretation layer that ties the whole project together:
combining RFM segment, churn risk (from the trained model), and CLV into a
single table that answers "who should get what kind of attention."

Design notes:
- The churn model was trained/validated on a 90-day-ago temporal holdout
  (see churn_features.py / churn_model.py) — that setup was specifically to
  get an honest, leakage-free ESTIMATE of model performance.
- For this final table, we want a churn-risk SCORE for every customer as of
  TODAY (not 90 days ago). So we recompute the same 7 features using the
  real present as the reference point, and apply the already-validated
  XGBoost model (refit on the full temporal dataset, since we're done
  evaluating and now want to use all available data for the actual scoring)
  to every current customer.
- Risk and Value are each split into 3 tiers (Low/Medium/High) using
  quantiles (tertiles), consistent with the percentile-based approach used
  throughout this project, giving a 3x3 = 9-cell action matrix.
"""

import sys
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, "src")
from rfm_analysis import compute_rfm
from clv_estimation import compute_clv
from churn_features import build_temporal_dataset

FEATURE_COLS = ["Recency", "Frequency", "Monetary", "tenure_days",
                "n_unique_products", "avg_order_value", "return_rate"]

ACTION_MAP = {
    ("High", "High"): "Urgent retention offer (personal outreach, discount, loyalty perk)",
    ("High", "Medium"): "Automated retention email / limited-time offer",
    ("High", "Low"): "Low-cost automated nudge only (not worth heavy investment)",
    ("Medium", "High"): "Proactive engagement (early access to new features/products)",
    ("Medium", "Medium"): "Standard engagement / newsletter",
    ("Medium", "Low"): "Low-touch, monitor only",
    ("Low", "High"): "Reward & retain (loyalty program, VIP perks, early access)",
    ("Low", "Medium"): "Maintain relationship / occasional check-in",
    ("Low", "Low"): "No action needed",
}


def compute_current_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature definitions as churn_features.py, but using the REAL
    present (last date in data + 1) as the reference point, over ALL
    transaction history — for scoring current customers, not for the
    leakage-free model validation."""
    reference_date = df.InvoiceDate.max() + pd.Timedelta(days=1)

    features = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineTotal", "sum"),
        first_purchase=("InvoiceDate", "min"),
        n_unique_products=("StockCode", "nunique"),
    )
    features["tenure_days"] = (reference_date - features.first_purchase).dt.days
    features["avg_order_value"] = features.Monetary / features.Frequency
    features.drop(columns=["first_purchase"], inplace=True)

    df = df.copy()
    df["is_cancellation"] = df.Invoice.astype(str).str.startswith("C")
    cancelled = df[df.is_cancellation].groupby("CustomerID").Invoice.nunique()
    total = df.groupby("CustomerID").Invoice.nunique()
    features["return_rate"] = (cancelled / total).reindex(features.index).fillna(0)

    return features


def build_segment_risk_value_table(df: pd.DataFrame) -> pd.DataFrame:
    rfm = compute_rfm(df)
    clv = compute_clv(df, rfm)

    # Train churn model on the full temporal (leakage-free) dataset
    temporal = build_temporal_dataset(df)
    X_train = temporal[FEATURE_COLS]
    y_train = temporal["is_churned"]
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # Score ALL current customers using today as the reference point
    current_features = compute_current_features(df)
    current_features = current_features.reindex(clv.index)  # align to same customer set
    churn_proba = model.predict_proba(current_features[FEATURE_COLS])[:, 1]

    table = clv.copy()
    table["churn_probability"] = churn_proba

    table["risk_tier"] = pd.qcut(
        table.churn_probability.rank(method="first"), 3, labels=["Low", "Medium", "High"]
    )
    table["value_tier"] = pd.qcut(
        table.historical_clv.rank(method="first"), 3, labels=["Low", "Medium", "High"]
    )
    table["recommended_action"] = table.apply(
        lambda row: ACTION_MAP[(row.risk_tier, row.value_tier)], axis=1
    )

    return table


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_transactions_retail.csv", parse_dates=["InvoiceDate"])
    table = build_segment_risk_value_table(df)

    print("Risk x Value cell counts:")
    print(pd.crosstab(table.risk_tier, table.value_tier))
    print()
    print("Sample of final table:")
    print(table[["Segment", "historical_clv", "churn_probability", "risk_tier",
                 "value_tier", "recommended_action"]].head(10))

    table.to_csv("data/processed/segment_risk_value_matrix.csv")
    print("\nSaved segment_risk_value_matrix.csv to data/processed/")