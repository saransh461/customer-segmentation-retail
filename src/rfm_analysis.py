"""
RFM (Recency, Frequency, Monetary) segmentation for the retail customer base.

Approach:
1. Compute raw Recency (days since last purchase), Frequency (distinct orders),
   and Monetary (total net spend) per customer.
2. Score each dimension 1-5 using quantiles (equal-sized buckets), so scores
   are directly comparable despite raw values living on very different scales.
   Recency is reversed (5 = most recent) since lower recency is better.
   Frequency/Monetary ties are broken with rank(method='first') before qcut,
   since many customers share low order counts and would otherwise break
   quantile binning.
3. Combine Frequency and Monetary scores into a single FM score (they're
   highly correlated — frequent buyers tend to spend more), giving a
   manageable 5x5 grid (Recency x FM) instead of a sparse 125-cell lookup.
4. Map each (R, FM) cell to a named, business-interpretable segment.
"""

import pandas as pd

SEGMENT_MAP = {
    (5, 5): "Champions", (5, 4): "Champions", (4, 5): "Champions",
    (5, 3): "Loyal Customers", (4, 4): "Loyal Customers", (4, 3): "Loyal Customers",
    (3, 5): "Loyal Customers", (3, 4): "Loyal Customers", (3, 3): "Loyal Customers",
    (5, 2): "Potential Loyalist", (4, 2): "Potential Loyalist",
    (5, 1): "Promising", (4, 1): "New Customers",
    (3, 2): "Need Attention", (3, 1): "About To Sleep",
    (2, 5): "Can't Lose Them", (1, 5): "Can't Lose Them",
    (2, 4): "At Risk", (2, 3): "At Risk", (1, 4): "At Risk", (1, 3): "At Risk",
    (2, 2): "Hibernating", (1, 2): "Hibernating",
    (2, 1): "Hibernating", (1, 1): "Lost",
}


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = df.InvoiceDate.max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineTotal", "sum"),
    )

    rfm["R_score"] = pd.qcut(rfm.Recency, 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm.Frequency.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm.Monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["FM_score"] = ((rfm.F_score + rfm.M_score) / 2).round().astype(int).clip(1, 5)

    rfm["Segment"] = rfm.apply(
        lambda row: SEGMENT_MAP.get((row.R_score, row.FM_score), "Unclassified"), axis=1
    )

    return rfm


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_transactions_retail.csv", parse_dates=["InvoiceDate"])
    rfm = compute_rfm(df)

    print("=== R/F/M correlation (justifies merging F and M into a single FM score) ===")
    print(rfm[["Recency", "Frequency", "Monetary"]].corr())
    print()

    print("Segment distribution:")
    print(rfm.Segment.value_counts())
    print()
    print(f"Unclassified customers (should be 0): {(rfm.Segment == 'Unclassified').sum()}")

    rfm.to_csv("data/processed/rfm_segments.csv")
    print("\nSaved rfm_segments.csv to data/processed/")