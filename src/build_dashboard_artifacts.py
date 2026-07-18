"""
Generates small, deployment-friendly artifacts for the Streamlit dashboard,
so the deployed app doesn't need the full 77MB transaction table or need to
retrain the churn model on every startup.

Run this once (and re-run after any pipeline change) to regenerate:
    python3 src/build_dashboard_artifacts.py
"""

import json
import sys
import pandas as pd

sys.path.insert(0, "src")


def build():
    df = pd.read_csv("data/processed/cleaned_transactions_retail.csv", parse_dates=["InvoiceDate"])

    monthly = df.set_index("InvoiceDate").resample("ME")["LineTotal"].sum()
    monthly.to_csv("data/processed/monthly_revenue.csv")

    summary = {
        "n_customers": int(df.CustomerID.nunique()),
        "n_transactions": int(df.Invoice.nunique()),
        "total_revenue": float(df.LineTotal.sum()),
        "date_range": f"{df.InvoiceDate.min().date()} to {df.InvoiceDate.max().date()}",
    }
    with open("data/processed/dashboard_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved monthly_revenue.csv and dashboard_summary.json to data/processed/")
    print(summary)


if __name__ == "__main__":
    build()