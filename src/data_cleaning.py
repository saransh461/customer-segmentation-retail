"""
Data cleaning pipeline for Online Retail II customer segmentation project.

Decisions made (documented for reproducibility):
1. Drop rows with missing CustomerID (~23% of rows, ~14% of revenue) — anonymous
   transactions can't be attributed to a customer, so they're excluded from
   customer-level analysis. Their volume/revenue share is logged as an EDA note.
2. Drop exact duplicate rows — known artifact of this dataset (double-logged entries).
3. Drop zero/negative price rows — small residual of manual adjustments/test
   products with no real transactional meaning (~70 rows).
4. Cancellations (Invoice starting with 'C') are KEPT, not dropped:
   - Their negative Quantity/LineTotal naturally nets out of a customer's total
     Monetary value when aggregated, giving true net spend.
   - A `return_rate` feature (cancelled invoices / total invoices) is engineered
     per customer as a churn-model signal.
"""

import pandas as pd


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="ISO-8859-1", parse_dates=["InvoiceDate"])
    df.rename(columns={"Customer ID": "CustomerID"}, inplace=True)

    raw_rows = len(df)

    # --- Document anonymous transactions before dropping ---
    anon = df[df.CustomerID.isna()]
    anon_revenue_share = (anon.Quantity * anon.Price).sum() / (df.Quantity * df.Price).sum()
    print(f"Anonymous rows: {len(anon)} ({len(anon)/raw_rows*100:.1f}% of all rows)")
    print(f"Anonymous revenue share: {anon_revenue_share*100:.1f}% of total revenue")

    df = df[df.CustomerID.notna()].copy()

    # --- Drop exact duplicates ---
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Dropped {dupes} exact duplicate rows")

    # --- Drop residual junk (zero/negative price, test products, manual adjustments) ---
    junk = df[df.Price <= 0]
    print(f"Dropped {len(junk)} zero/negative price rows (manual adjustments, test entries)")
    df = df[df.Price > 0].copy()

    # --- Flag cancellations, compute line-level net revenue ---
    df["is_cancellation"] = df.Invoice.astype(str).str.startswith("C")
    df["LineTotal"] = df.Quantity * df.Price

    print(f"Final cleaned shape: {df.shape}")
    print(f"Unique customers: {df.CustomerID.nunique()}")
    print(f"Unique invoices: {df.Invoice.nunique()}")

    return df


def compute_return_rate(df: pd.DataFrame) -> pd.Series:
    """Per-customer return rate: cancelled invoices / total invoices."""
    cancelled_per_cust = df[df.is_cancellation].groupby("CustomerID").Invoice.nunique()
    total_per_cust = df.groupby("CustomerID").Invoice.nunique()
    return (cancelled_per_cust / total_per_cust).fillna(0).rename("return_rate")


if __name__ == "__main__":
    cleaned = load_and_clean("data/raw/online_retail_II.csv")
    return_rate = compute_return_rate(cleaned)

    cleaned.to_csv("data/processed/cleaned_transactions.csv", index=False)
    return_rate.to_csv("data/processed/customer_return_rate.csv")
    print("\nSaved cleaned_transactions.csv and customer_return_rate.csv to data/processed/")