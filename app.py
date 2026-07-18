"""
Lightweight Streamlit dashboard for the customer segmentation project.

Design change from the first version: this app now reads from small,
PRE-COMPUTED artifacts (segment_risk_value_matrix.csv, monthly_revenue.csv,
dashboard_summary.json — all under 1MB combined) instead of the full
794,183-row transaction table (77MB) and instead of retraining XGBoost on
every app startup.

Why: the full transaction file is too large to comfortably commit to GitHub
(same reasoning as excluding the raw 95MB dataset earlier), and Streamlit
Cloud only has access to what's actually in the repo. Precomputing the
customer-level outputs once (via the existing pipeline scripts) and shipping
just those small results is both a deployment fix AND better practice in
general — the deployed app doesn't need to redo expensive model training on
every cold start just to display results that don't change between runs.

To regenerate these artifacts after any pipeline change, run:
    python3 src/segment_risk_value.py
    python3 src/build_dashboard_artifacts.py
"""

import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")


@st.cache_data
def load_data():
    table = pd.read_csv("data/processed/segment_risk_value_matrix.csv", index_col="CustomerID")
    monthly_revenue = pd.read_csv("data/processed/monthly_revenue.csv", index_col="InvoiceDate", parse_dates=True)
    with open("data/processed/dashboard_summary.json") as f:
        summary = json.load(f)
    return table, monthly_revenue, summary


table, monthly_revenue, summary = load_data()

st.title("Customer Segmentation, Churn Risk & CLV Dashboard")
st.caption("Online Retail II dataset (Dec 2009 - Dec 2011) — retail customers only, wholesale accounts analyzed separately")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Segment Explorer", "Customer Lookup", "Business Insights"]
)

# --- TAB 1: OVERVIEW ---
with tab1:
    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{summary['n_customers']:,}")
    col2.metric("Transactions", f"{summary['n_transactions']:,}")
    col3.metric("Total Revenue", f"£{summary['total_revenue']:,.0f}")
    col4.metric("Date Range", summary['date_range'])

    st.markdown("""
    **Key data quality decisions made during cleaning:**
    - Dropped anonymous transactions (23% of rows, 14% of revenue) — can't be attributed to a customer
    - Dropped exact duplicates and non-product line items (postage, fees, manual adjustments)
    - Split off 68 wholesale-pattern accounts (bulk buyers, 17% of revenue) for separate analysis
    - Kept cancellations, netted into spend + engineered as a `return_rate` feature
    """)

    st.subheader("Monthly Revenue Trend")
    st.line_chart(monthly_revenue["LineTotal"])

# --- TAB 2: SEGMENT EXPLORER ---
with tab2:
    st.subheader("RFM Segment Distribution")
    seg_counts = table.Segment.value_counts()
    st.bar_chart(seg_counts)

    st.subheader("Segment Details")
    segment_summary = table.groupby("Segment").agg(
        customers=("Recency", "count"),
        avg_recency=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
    ).round(1).sort_values("customers", ascending=False)
    st.dataframe(segment_summary, use_container_width=True)

# --- TAB 3: CUSTOMER LOOKUP ---
with tab3:
    st.subheader("Look Up a Customer")
    customer_id = st.selectbox("Select Customer ID", sorted(table.index.astype(int)))

    row = table.loc[float(customer_id)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Segment", row.Segment)
    col2.metric("Historical CLV", f"£{row.historical_clv:,.0f}")
    col3.metric("Churn Probability", f"{row.churn_probability*100:.0f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("Risk Tier", row.risk_tier)
    col5.metric("Value Tier", row.value_tier)
    col6.metric("Recency (days)", int(row.Recency))

    st.info(f"**Recommended action:** {row.recommended_action}")

# --- TAB 4: BUSINESS INSIGHTS ---
with tab4:
    st.subheader("Risk x Value Action Matrix")
    st.caption("Each cell shows customer count and average historical CLV")

    counts = pd.crosstab(table.risk_tier, table.value_tier)
    avg_clv = table.pivot_table(index="risk_tier", columns="value_tier",
                                  values="historical_clv", aggfunc="mean", observed=True).round(0)

    display = counts.astype(str) + " customers (avg CLV £" + avg_clv.astype(str) + ")"
    st.dataframe(display, use_container_width=True)

    st.subheader("Recommended Actions by Cell")
    action_summary = table.groupby(["risk_tier", "value_tier"], observed=True).agg(
        customers=("Recency", "count"),
        avg_clv=("historical_clv", "mean"),
        recommended_action=("recommended_action", "first"),
    ).round(0).reset_index()
    st.dataframe(action_summary, use_container_width=True)