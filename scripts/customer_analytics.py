
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Fixed snapshot date required by the capstone
SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cleaned data
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # Only completed orders count towards customer behaviour
    completed = orders[
        orders["status"].str.strip().str.lower() == "completed"
    ].copy()

    # Calculate RFM for every customer with completed orders
    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    ).reset_index()

    # Recency is the whole number of days since the last completed order
    rfm["recency"] = (
        SNAPSHOT_DATE - rfm["last_order_ts"]
    ).dt.days

    # Keep the requested RFM columns
    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ]

    # Round monetary values to two decimal places
    rfm["monetary"] = rfm["monetary"].round(2)

    # Save the RFM table
    output_path = OUTPUT_DIR / "rfm_table.csv"
    rfm.to_csv(output_path, index=False)
    
    # Standardise RFM values before clustering
    rfm_features = rfm[["recency", "frequency", "monetary"]]

    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm_features)

    # Test K values from 2 to 8
    clustering_results = []

    print("\nK-MEANS CLUSTER EVALUATION")
    print("=" * 55)

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )

        labels = model.fit_predict(scaled_rfm)

        inertia = model.inertia_
        silhouette = silhouette_score(scaled_rfm, labels)

        clustering_results.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(silhouette, 3),
        })

    clustering_df = pd.DataFrame(clustering_results)

    print(clustering_df.to_string(index=False))

    clustering_path = OUTPUT_DIR / "kmeans_evaluation.csv"
    clustering_df.to_csv(clustering_path, index=False)

    print("\nSaved clustering evaluation to:", clustering_path)


    # Display summary statistics
    print("RFM TABLE CHECK")
    print("=" * 40)
    print("Customers in RFM table:", len(rfm))
    print("Median recency:", rfm["recency"].median())
    print("Minimum recency:", rfm["recency"].min())
    print("Maximum recency:", rfm["recency"].max())
    print("Mean frequency:", round(rfm["frequency"].mean(), 2))
    print("Maximum frequency:", rfm["frequency"].max())
    print("Mean monetary:", round(rfm["monetary"].mean(), 2))
    print("Maximum monetary:", rfm["monetary"].max())
    print("Saved to:", output_path)
        # Create the final customer segments using K = 4
    final_model = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=42
    )

    rfm["segment_id"] = final_model.fit_predict(scaled_rfm)

    # Profile each segment
    segment_profile = rfm.groupby("segment_id").agg(
        customer_count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).round(2)

    print("\nFINAL CUSTOMER SEGMENT PROFILE")
    print("=" * 60)
    print(segment_profile.to_string())

    # Identify the segment with the highest average spending
    premium_segment = segment_profile["mean_monetary"].idxmax()

    # Give each segment a readable name
    rfm["segment_name"] = rfm["segment_id"].apply(
        lambda segment: (
            "Premium" if segment == premium_segment
            else f"Segment {segment}"
        )
    )

    rfm["is_premium"] = (
        rfm["segment_id"] == premium_segment
    ).astype(int)

    # Save the final customer segmentation
    segments_path = OUTPUT_DIR / "customer_segments.csv"
    rfm.to_csv(segments_path, index=False)

    print("\nPremium segment ID:", premium_segment)
    print("Premium customer count:", int(rfm["is_premium"].sum()))
    print("Saved customer segments to:", segments_path)
        # Reduce the three RFM features to two dimensions using PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(scaled_rfm)

    explained_variance = pca.explained_variance_ratio_

    print("\nPCA EXPLAINED VARIANCE")
    print("=" * 40)
    print("Component 1:", round(explained_variance[0], 3))
    print("Component 2:", round(explained_variance[1], 3))
    print(
        "Combined:",
        round(explained_variance.sum(), 3)
    )


if __name__ == "__main__":
    main()
