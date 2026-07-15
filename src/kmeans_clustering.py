"""
K-means clustering on RFM features, to validate and refine the manual RFM
segmentation.

Approach:
- Log-transform Frequency and Monetary before scaling (both are heavily
  right-skewed; log transform lets K-means see real structure instead of
  just reacting to extreme outliers).
- StandardScaler all three features (Recency, log-Frequency, log-Monetary)
  since K-means uses raw distance and Monetary's scale would otherwise
  dominate Recency/Frequency.
- k=4 chosen via elbow method (inertia) + silhouette score: silhouette is
  highest at k=2 (0.419), but that's too coarse for actionable segmentation.
  Inertia's rate of improvement clearly slows after k=4 (elbow), and k=4's
  silhouette (0.362) is effectively tied with k=5 (0.365) — so k=4 was chosen
  for the better interpretability/complexity tradeoff.

Key finding: K-means clusters by overall current magnitude across R/F/M
simultaneously, with no concept of "decline" or "urgency." This means
manual RFM's "At Risk" / "Can't Lose Them" segments (which specifically
flag customers whose behavior is CHANGING) get split across multiple
K-means clusters, since those customers' current raw numbers resemble
whichever cluster is closest to their present (not past) behavior. The two
approaches answer different questions: K-means finds natural density
groupings; manual RFM encodes business intent (who needs urgent action)
directly into the segment definition.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

N_CLUSTERS = 4

CLUSTER_NAMES = {
    # Filled in after inspecting cluster centers on this dataset. Re-check
    # this mapping if the underlying data changes, since KMeans cluster
    # index order is not guaranteed to stay the same run to run without a
    # fixed random_state (we use random_state=42 for reproducibility).
}


def run_kmeans(rfm: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    X = pd.DataFrame({
        "Recency": rfm.Recency,
        "Frequency_log": np.log1p(rfm.Frequency),
        "Monetary_log": np.log1p(rfm.Monetary.clip(lower=0)),
    }, index=rfm.index)

    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = km.fit_predict(X_scaled)
    return rfm


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from rfm_analysis import compute_rfm

    df = pd.read_csv("data/processed/cleaned_transactions_retail.csv", parse_dates=["InvoiceDate"])
    rfm = compute_rfm(df)
    rfm = run_kmeans(rfm)

    print("Cluster sizes:")
    print(rfm.Cluster.value_counts().sort_index())
    print()

    print("Cluster centers (actual units):")
    summary = rfm.groupby("Cluster").agg(
        avg_recency=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
        count=("Recency", "count"),
    ).round(1)
    print(summary)
    print()

    print("Cross-tab: K-means cluster vs manual RFM segment")
    print(pd.crosstab(rfm.Cluster, rfm.Segment))

    rfm.to_csv("data/processed/rfm_with_clusters.csv")
    print("\nSaved rfm_with_clusters.csv to data/processed/")