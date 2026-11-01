
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CUTOFF_DATE = pd.Timestamp("2025-09-01")
LABEL_END_DATE = pd.Timestamp("2025-10-01")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cleaned data
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    orders["status"] = orders["status"].str.strip().str.lower()

    # Completed orders before the cutoff
    historical = orders[
        (orders["status"] == "completed")
        & (orders["order_ts"] < CUTOFF_DATE)
    ].copy()

    # Customers who qualify for the prediction cohort
    cohort_ids = historical["customer_id"].unique()
    cohort = customers[
        customers["customer_id"].isin(cohort_ids)
    ].copy()

    # Completed orders during September determine the label
    september_orders = orders[
        (orders["status"] == "completed")
        & (orders["order_ts"] >= CUTOFF_DATE)
        & (orders["order_ts"] < LABEL_END_DATE)
    ]

    september_customer_ids = set(
        september_orders["customer_id"].unique()
    )

    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(september_customer_ids)
    ).astype(int)

    # Calculate historical features using only data before cutoff
    historical_features = historical.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    ).reset_index()

    historical_features["recency"] = (
        CUTOFF_DATE - historical_features["last_order_ts"]
    ).dt.days

    # Historical cancellation rate: cancelled / all orders before cutoff
    all_historical = orders[
        orders["order_ts"] < CUTOFF_DATE
    ].copy()

    cancellation_rate = all_historical.groupby("customer_id").agg(
        total_orders=("order_id", "nunique"),
        cancelled_orders=(
            "status",
            lambda values: (values == "cancelled").sum()
        ),
    ).reset_index()

    cancellation_rate["cancellation_rate"] = (
        cancellation_rate["cancelled_orders"]
        / cancellation_rate["total_orders"]
    )

    # Most frequently ordered product category before cutoff
    products = pd.read_csv(CLEAN_DIR / "products.csv")

    historical_with_category = historical.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    favourite_category = (
        historical_with_category
        .groupby("customer_id")["category"]
        .agg(lambda values: values.mode().iloc[0])
        .rename("favourite_category")
        .reset_index()
    )

    # Combine features with cohort labels
    dataset = (
        cohort
        .merge(historical_features, on="customer_id", how="left")
        .merge(
            cancellation_rate[["customer_id", "cancellation_rate"]],
            on="customer_id",
            how="left"
        )
        .merge(favourite_category, on="customer_id", how="left")
    )

    dataset["tenure_days"] = (
        CUTOFF_DATE - pd.to_datetime(dataset["signup_date"])
    ).dt.days

    dataset["avg_discount"] = dataset["avg_discount"].fillna(0)
    dataset["avg_delivery_minutes"] = (
        dataset["avg_delivery_minutes"].fillna(
            dataset["avg_delivery_minutes"].median()
        )
    )
    dataset["cancellation_rate"] = (
        dataset["cancellation_rate"].fillna(0)
    )

    # Save dataset
    output_path = OUTPUT_DIR / "lapse_dataset.csv"
    dataset.to_csv(output_path, index=False)

    print("LAPSE DATASET CHECK")
    print("=" * 40)
    print("Cohort customers:", len(dataset))
    print("Lapsed customers:", int(dataset["lapsed"].sum()))
    print(
        "Lapse rate (%):",
        round(dataset["lapsed"].mean() * 100, 2)
    )
    print("Missing values:")
    print(dataset.isna().sum().to_string())
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
