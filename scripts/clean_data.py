
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"


def audit_table(name, dataframe):
    print(f"\n{'=' * 50}")
    print(f"AUDIT: {name}")
    print(f"{'=' * 50}")

    print(f"\nShape: {dataframe.shape}")

    print("\nData types:")
    print(dataframe.dtypes.to_string())

    print("\nMissing values by column:")
    audit = pd.DataFrame({
        "missing_count": dataframe.isna().sum(),
        "missing_percent": (
            dataframe.isna().mean() * 100
        ).round(2),
    })
    print(audit.to_string())


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(RAW_DIR / "customers.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")
    orders = pd.read_csv(RAW_DIR / "orders.csv")

    # Clean order amounts
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )


    # Fill missing order amounts using product price and order details
    product_prices = products.set_index("product_id")["price_inr"]

    missing_amounts = orders["order_amount"].isna()

    orders.loc[missing_amounts, "order_amount"] = (
        orders.loc[missing_amounts, "product_id"].map(product_prices)
        * orders.loc[missing_amounts, "quantity"]
        * (
            1 - orders.loc[missing_amounts, "discount_pct"] / 100
        )
    ).round(2)

    print(
        "Missing order amounts after filling:",
        orders["order_amount"].isna().sum()
    )

    # Remove exact duplicate rows
    duplicate_count = orders.duplicated().sum()
    orders = orders.drop_duplicates().copy()

    # Convert order timestamps to datetime
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    # Keep the latest record for each order ID
    orders = orders.sort_values("order_ts")
    orders = orders.drop_duplicates(
        subset="order_id", keep="last"
    ).copy()

    print("Exact duplicate rows removed:", duplicate_count)
    print("Orders remaining after latest-record rule:", len(orders))
    print("Duplicate order IDs remaining:", orders["order_id"].duplicated().sum())


    
    
    # Clean delivery times
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    negative_delivery_count = orders["delivery_minutes"].lt(0).sum()

    orders.loc[
        orders["delivery_minutes"] < 0,
        "delivery_minutes"
    ] = float("nan")

    print(
        "Negative delivery times replaced with missing:",
        negative_delivery_count
    )

    # Find the category of each product
    category_lookup = products.set_index("product_id")["category"]

    # Identify completed orders
    completed_mask = (
        orders["status"].str.strip().str.lower() == "completed"
    )

    # Calculate median delivery time for each category,
    # using completed orders only
    completed_category_medians = (
        orders.loc[completed_mask]
        .groupby(
            orders.loc[completed_mask, "product_id"].map(category_lookup)
        )["delivery_minutes"]
        .median()
    )

    # Fill missing delivery times for completed orders only
    missing_completed = (
        completed_mask & orders["delivery_minutes"].isna()
    )

    orders.loc[missing_completed, "delivery_minutes"] = (
        orders.loc[missing_completed, "product_id"]
        .map(category_lookup)
        .map(completed_category_medians)
    )

    # Check the results
    cancelled_mask = (
        orders["status"].str.strip().str.lower() == "cancelled"
    )

    print(
        "Completed orders with missing delivery times:",
        orders.loc[completed_mask, "delivery_minutes"].isna().sum()
    )

    print(
        "Cancelled orders with missing delivery times:",
        orders.loc[cancelled_mask, "delivery_minutes"].isna().sum()
    )

    print(
        "Total missing delivery times:",
        orders["delivery_minutes"].isna().sum()
    )

    # Convert order timestamps to datetime
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    print(
        "Invalid order timestamps:",
        orders["order_ts"].isna().sum()
    )
    print("\nORDER CLEANING CHECK")
    print("Exact duplicate rows removed:", duplicate_count)
    print("Orders remaining:", len(orders))

    # Clean customer data
    customers["city"] = customers["city"].str.strip().str.title()

    gender_map = {
        "m": "male",
        "male": "male",
        "man": "male",
        "men": "male",
        "f": "female",
        "female": "female",
        "woman": "female",
        "women": "female",
    }

    customers["gender"] = (
        customers["gender"]
        .fillna("")
        .str.strip()
        .str.lower()
        .map(gender_map)
        .fillna("unknown")
    )

    customers["age"] = pd.to_numeric(
        customers["age"], errors="coerce"
    )
    customers.loc[
        ~customers["age"].between(10, 100),
        "age"
    ] = float("nan")

    city_median_age = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_median_age)

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    print("\nCUSTOMER CLEANING CHECK")
    print("Cities:", sorted(customers["city"].unique()))
    print("Gender values:", sorted(customers["gender"].unique()))
    print("Missing ages after cleaning:", customers["age"].isna().sum())
    print("Signup date type:", customers["signup_date"].dtype)

    audit_table("Customers", customers)
    audit_table("Products", products)
    audit_table("Orders", orders)

    # Required data-quality checks
    print("\n" + "=" * 50)
    print("REQUIRED DATA-QUALITY CHECKS")
    print("=" * 50)

    numeric_amounts = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    numeric_delivery = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    print("\nORDERS")
    print("Raw order rows:", len(orders))
    print("Exact duplicate rows:", orders.duplicated().sum())
    print(
        "Pending rows:",
        orders["status"].str.strip().str.lower().eq("pending").sum()
    )
    print("Distinct order IDs:", orders["order_id"].nunique())
    print("Non-numeric order amounts:", numeric_amounts.isna().sum())
    print("Negative delivery times:", numeric_delivery.lt(0).sum())
    print(
        "Blank delivery times:",
        orders["delivery_minutes"].isna().sum()
    )

    print("\nCUSTOMERS")
    print("Blank ages:", customers["age"].isna().sum())
    print(
        "Ages outside 10-100:",
        (~customers["age"].between(10, 100)).sum()
        - customers["age"].isna().sum()
    )
    print("Distinct raw city spellings:", customers["city"].nunique())
    print(
        "Distinct non-blank gender spellings:",
        customers.loc[
            customers["gender"].notna()
            & customers["gender"].str.strip().ne(""),
            "gender"
        ].nunique()
    )
    print(
        "Blank genders:",
        (
            customers["gender"].isna()
            | customers["gender"].str.strip().eq("")
        ).sum()
    )

    print("\nData-quality checks completed.")

    print("\nRaw data audit completed.")
    # Save cleaned customer data
    customers.to_csv(
        CLEAN_DIR / "customers.csv",
        index=False,
        date_format="%Y-%m-%d"
    )
    
    # Clean and save products
    for column in ["product_id", "product_name", "category"]:
        products[column] = products[column].str.strip()

    products["price_inr"] = pd.to_numeric(
        products["price_inr"], errors="raise"
    )
    products.to_csv(
        CLEAN_DIR / "products.csv",
        index=False
    )

    # Save cleaned order data
    orders.to_csv(
        CLEAN_DIR / "orders.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S"
    )
    print("Cleaned orders saved to data/clean/orders.csv")

    print("Cleaned products saved to data/clean/products.csv")


    print("\nCleaned customers saved to data/clean/customers.csv")

if __name__ == "__main__":
    main()
