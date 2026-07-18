"""
Lightweight Streamlit dashboard for the customer segmentation project.

Structure (4 simple sections, no heavy custom UI):
1. Overview - dataset summary
2. Segment Explorer - RFM segment distribution
3. Customer Lookup - look up any customer's segment/risk/value/action
4. Business Insights - the Risk x Value action matrix (the project's payoff)

Design choice: this app calls the existing pipeline functions directly
(compute_rfm, compute_clv, build_segment_risk_value_table) rather than
duplicating logic, so the dashboard and the underlying analysis never
drift out of sync. Everything is wrapped in st.cache_data / st.cache_resource
so the (somewhat expensive) XGBoost training only happens once per session,
not on every interaction.
"""

import sys
sys.path.insert(0, "src")

import pandas as pd
import streamlit as st

from rfm_analysis import compute_rfm
from clv_estimation import compute_clv
from segment_risk_value import build_segment_risk_value_table

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")


@st.cache_data
def load_raw_data():
    return pd.read_csv("data/processed/cleaned_transactions_retail.csv", parse_dates=["InvoiceDate"])


@st.cache_data
def load_full_table(_df):
    return build_segment_risk_value_table(_df)


df = load_raw_data()
table = load_full_table(df)

st.title("Customer Segmentation, Churn Risk & CLV Dashboard")
st.caption("Online Retail II dataset (Dec 2009 - Dec 2011) — retail customers only, wholesale accounts analyzed separately")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Segment Explorer", "Customer Lookup", "Business Insights"]
)

# --- TAB 1: OVERVIEW ---
with tab1:
    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{df.CustomerID.nunique():,}")
    col2.metric("Transactions", f"{df.Invoice.nunique():,}")
    col3.metric("Total Revenue", f"£{df.LineTotal.sum():,.0f}")
    col4.metric("Date Range", "2 years")

    st.markdown("""
    **Key data quality decisions made during cleaning:**
    - Dropped anonymous transactions (23% of rows, 14% of revenue) — can't be attributed to a customer
    - Dropped exact duplicates and non-product line items (postage, fees, manual adjustments)
    - Split off 68 wholesale-pattern accounts (bulk buyers, 17% of revenue) for separate analysis
    - Kept cancellations, netted into spend + engineered as a `return_rate` feature
    """)

    st.subheader("Monthly Revenue Trend")
    monthly = df.set_index("InvoiceDate").resample("ME")["LineTotal"].sum()
    st.line_chart(monthly)

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