"""
Time-based (leakage-free) feature engineering for churn prediction.

Design (see conversation history for full reasoning):
- Cutoff date = last date in dataset minus 90 days (~Sept 10, 2011).
  This simulates "standing at an earlier point in time."
- FEATURES are computed using ONLY transactions on/before the cutoff —
  exactly what would have been knowable at that point in time.
- The LABEL is computed using ONLY transactions AFTER the cutoff: did the
  customer make any purchase in the 90 days following the cutoff? If not,
  they're labeled churned.
- The modeling population is restricted to customers who had already made
  at least one purchase before the cutoff (a customer who first appears
  after the cutoff isn't a real prediction case — they didn't exist yet at
  "prediction time").

This avoids the leakage trap of using Recency (measured at the dataset's
final day) to predict a Recency-based churn label — which would be circular,
since the model would just be restating its own label definition rather than
forecasting genuinely unseen future behavior.
"""

import pandas as pd

HOLD_OUT_DAYS = 90


def build_temporal_dataset(df: pd.DataFrame, hold_out_days: int = HOLD_OUT_DAYS) -> pd.DataFrame:
    max_date = df.InvoiceDate.max()
    cutoff = max_date - pd.Timedelta(days=hold_out_days)

    pre = df[df.InvoiceDate <= cutoff].copy()
    post = df[df.InvoiceDate > cutoff].copy()

    existing_customers = pre.CustomerID.unique()
    pre = pre[pre.CustomerID.isin(existing_customers)]

    # --- Features: computed ONLY from pre-cutoff data ---
    features = pre.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineTotal", "sum"),
        first_purchase=("InvoiceDate", "min"),
        n_unique_products=("StockCode", "nunique"),
    )
    features["tenure_days"] = (cutoff - features.first_purchase).dt.days
    features["avg_order_value"] = features.Monetary / features.Frequency
    features.drop(columns=["first_purchase"], inplace=True)

    # Return rate (pre-cutoff only)
    pre["is_cancellation"] = pre.Invoice.astype(str).str.startswith("C")
    cancelled_per_cust = pre[pre.is_cancellation].groupby("CustomerID").Invoice.nunique()
    total_per_cust = pre.groupby("CustomerID").Invoice.nunique()
    features["return_rate"] = (cancelled_per_cust / total_per_cust).reindex(features.index).fillna(0)

    # --- Label: computed ONLY from post-cutoff data ---
    returned_customers = set(post[post.CustomerID.isin(existing_customers)].CustomerID.unique())
    features["is_churned"] = (~features.index.isin(returned_customers)).astype(int)

    return features


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_transactions_retail.csv", parse_dates=["InvoiceDate"])
    dataset = build_temporal_dataset(df)

    print(f"Modeling population: {len(dataset)} customers")
    print()
    print("Feature summary:")
    print(dataset.describe())
    print()
    print("Label distribution:")
    print(dataset.is_churned.value_counts())
    print(f"Churn rate: {dataset.is_churned.mean()*100:.1f}%")

    dataset.to_csv("data/processed/churn_model_dataset.csv")
    print("\nSaved churn_model_dataset.csv to data/processed/")