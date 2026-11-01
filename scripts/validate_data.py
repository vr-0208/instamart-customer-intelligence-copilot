
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"


def main():
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")

    checks = [
        ("Customer count is 1200", len(customers) == 1200),
        ("Product count is 40", len(products) == 40),
        ("Order count is 12309", len(orders) == 12309),
        ("Customer IDs are unique", customers["customer_id"].is_unique),
        ("Product IDs are unique", products["product_id"].is_unique),
        ("Order IDs are unique", orders["order_id"].is_unique),
        ("No missing customer ages", customers["age"].notna().all()),
        ("No missing order amounts", orders["order_amount"].notna().all()),
        (
            "No negative delivery times",
            (orders["delivery_minutes"].dropna() >= 0).all()
        ),
        (
            "Completed orders have delivery times",
            orders.loc[
                orders["status"].str.lower() == "completed",
                "delivery_minutes"
            ].notna().all()
        ),
        (
            "Cancelled orders have missing delivery times",
            orders.loc[
                orders["status"].str.lower() == "cancelled",
                "delivery_minutes"
            ].isna().all()
        ),
        (
            "No pending orders remain",
            not orders["status"].str.lower().eq("pending").any()
        ),
    ]

    failed = []

    print("\nDATA VALIDATION RESULTS")
    print("=" * 40)

    for description, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {description}")
        if not passed:
            failed.append(description)

    if failed:
        raise AssertionError(
            f"{len(failed)} validation check(s) failed."
        )

    print("\nALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
