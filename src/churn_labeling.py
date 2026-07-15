"""
Churn labeling for the retail customer base.

Threshold: 90 days since last purchase = churned.

Justification (not an arbitrary guess):
- Computed the distribution of gaps between consecutive orders across all
  customers (36,885 gaps total).
- Median gap: 16 days. 75th percentile: 49 days. 90th percentile: 118 days.
- 90 days sits between the 75th and 90th percentile of genuine repeat-purchase
  behavior — i.e. it's longer than how long ~75-85% of active customers
  naturally wait between orders, so silence beyond this point is atypical
  enough to flag as churn risk, while still being early enough (~3 months)
  to act on with a retention offer.
- See notebooks/eda.ipynb (or reports/) for the full repurchase-gap histogram.
"""

import pandas as pd

CHURN_THRESHOLD_DAYS = 90


def compute_churn_label(df: pd.DataFrame, threshold: int = CHURN_THRESHOLD_DAYS) -> pd.DataFrame:
    reference_date = df.InvoiceDate.max() + pd.Timedelta(days=1)

    last_purchase = df.groupby("CustomerID").InvoiceDate.max()
    recency = (reference_date - last_purchase).dt.days

    churn = pd.DataFrame({
        "Recency": recency,
        "is_churned": recency > threshold
    })

    return churn


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_transactions_retail.csv", parse_dates=["InvoiceDate"])
    churn = compute_churn_label(df)

    print(f"Churn threshold: {CHURN_THRESHOLD_DAYS} days")
    print()
    print("Churn label distribution:")
    print(churn.is_churned.value_counts())
    print(f"\nChurn rate: {churn.is_churned.mean()*100:.1f}%")

    churn.to_csv("data/processed/churn_labels.csv")
    print("\nSaved churn_labels.csv to data/processed/")