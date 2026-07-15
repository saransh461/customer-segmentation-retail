"""
Customer Lifetime Value (CLV) estimation.

Two complementary CLV numbers are computed, deliberately kept separate rather
than blended into one figure, since they answer different questions:

1. Historical CLV = total net spend to date (same as RFM's Monetary).
   Backward-looking, no assumptions, always defensible.

2. Projected annual value = (Historical CLV / tenure_days) * 365.
   Forward-looking estimate of "value per year" based on their observed
   spending rate. ONLY computed for customers with tenure_days >= 30 —
   a customer with only a few days of observed history can produce wildly
   inflated projections (e.g. one customer with 1 day of tenure and modest
   spend projected to ~₹410,000/year despite being 597 days recency-silent).
   A 30-day minimum window was chosen as a practical floor for a rate
   estimate to be considered even roughly reliable. Customers below this
   threshold are explicitly flagged as `has_sufficient_history = False`
   rather than silently given a misleading projected number.
"""

import numpy as np
import pandas as pd

MIN_TENURE_FOR_PROJECTION = 30


def compute_clv(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    clv = rfm.copy()

    tenure = df.groupby("CustomerID").InvoiceDate.agg(["min", "max"])
    clv["tenure_days"] = (tenure["max"] - tenure["min"]).dt.days

    clv["historical_clv"] = clv.Monetary

    clv["has_sufficient_history"] = clv.tenure_days >= MIN_TENURE_FOR_PROJECTION
    clv["projected_annual_value"] = np.where(
        clv.has_sufficient_history,
        (clv.historical_clv / clv.tenure_days) * 365,
        np.nan,
    )

    return clv


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from rfm_analysis import compute_rfm

    df = pd.read_csv("data/processed/cleaned_transactions_retail.csv", parse_dates=["InvoiceDate"])
    rfm = compute_rfm(df)
    clv = compute_clv(df, rfm)

    print(f"Customers with sufficient history for projection: "
          f"{clv.has_sufficient_history.sum()} ({clv.has_sufficient_history.mean()*100:.1f}%)")
    print()
    print("Historical CLV distribution:")
    print(clv.historical_clv.describe(percentiles=[.5, .75, .9, .95, .99]))
    print()
    print("Projected annual value distribution (only where computable):")
    print(clv.projected_annual_value.describe(percentiles=[.5, .75, .9, .95, .99]))

    clv.to_csv("data/processed/customer_clv.csv")
    print("\nSaved customer_clv.csv to data/processed/")