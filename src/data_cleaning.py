"""
Data cleaning pipeline for Online Retail II customer segmentation project.

Decisions made (documented for reproducibility):
1. Drop rows with missing CustomerID (~23% of rows, ~14% of revenue) — anonymous
   transactions can't be attributed to a customer, so they're excluded from
   customer-level analysis. Their volume/revenue share is logged as an EDA note.
2. Drop exact duplicate rows — known artifact of this dataset (double-logged entries).
3. Drop zero/negative price rows — small residual of manual adjustments/test
   products with no real transactional meaning (~70 rows).
4. Drop non-product StockCodes (POST, DOT, M, C2, D, ADJUST, ADJUST2,
   BANK CHARGES, CRUK, TEST001, TEST002) — these are postage, fees, discounts,
   and manual adjustments logged in the same table as real products, and would
   distort Monetary/product-level analysis if left in (~3,641 rows, ~19% of
   customers affected). Note: codes like 15056BL, 79323LP, PADS, SP1002 look
   non-standard but ARE real products — confirmed via Description before
   excluding anything, rather than relying on a blanket regex.
5. Cancellations (Invoice starting with 'C') are KEPT, not dropped:
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

    # --- Drop non-product StockCodes (postage, fees, discounts, manual adjustments) ---
    # Confirmed via Description that these specific codes are non-product entries;
    # similar-looking codes (e.g. 15056BL, 79323LP, PADS) were checked and kept
    # since they are real products with non-standard code formats.
    non_product_codes = [
        "POST", "DOT", "M", "C2", "D", "ADJUST", "ADJUST2",
        "BANK CHARGES", "CRUK", "TEST001", "TEST002",
    ]
    non_product_rows = df.StockCode.isin(non_product_codes)
    print(f"Dropped {non_product_rows.sum()} non-product rows (postage/fees/discounts/adjustments)")
    df = df[~non_product_rows].copy()

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


def flag_wholesale(df: pd.DataFrame, threshold: float = 1000) -> pd.Series:
    """
    Flags customers with wholesale-like buying patterns, based on average
    quantity purchased per invoice. Threshold of 1000 was chosen because it
    sits almost exactly at the 99th percentile of this metric across all
    customers (measured: ~1043.5), marking a genuine behavioral break rather
    than an arbitrary cutoff — the median customer orders ~132 units/invoice,
    while flagged accounts order in the thousands to tens of thousands.
    """
    cust_qty = df.groupby("CustomerID").apply(
        lambda g: g.Quantity.sum() / g.Invoice.nunique(), include_groups=False
    )
    return (cust_qty > threshold).rename("is_wholesale")


if __name__ == "__main__":
    cleaned = load_and_clean("data/raw/online_retail_II.csv")
    return_rate = compute_return_rate(cleaned)
    wholesale_flag = flag_wholesale(cleaned)

    n_wholesale = wholesale_flag.sum()
    wholesale_ids = wholesale_flag[wholesale_flag].index
    print(f"\nFlagged {n_wholesale} wholesale-pattern customers "
          f"({n_wholesale/cleaned.CustomerID.nunique()*100:.2f}% of customers)")

    retail_df = cleaned[~cleaned.CustomerID.isin(wholesale_ids)].copy()
    wholesale_df = cleaned[cleaned.CustomerID.isin(wholesale_ids)].copy()

    retail_df.to_csv("data/processed/cleaned_transactions_retail.csv", index=False)
    wholesale_df.to_csv("data/processed/cleaned_transactions_wholesale.csv", index=False)
    return_rate.to_csv("data/processed/customer_return_rate.csv")

    print(f"Retail dataset: {retail_df.shape}, {retail_df.CustomerID.nunique()} customers")
    print(f"Wholesale dataset: {wholesale_df.shape}, {wholesale_df.CustomerID.nunique()} customers")
    print("\nSaved cleaned_transactions_retail.csv, cleaned_transactions_wholesale.csv, "
          "and customer_return_rate.csv to data/processed/")