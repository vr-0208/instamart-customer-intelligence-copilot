
from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"


def main():
    # Read completed orders from SQLite
    with sqlite3.connect(DB_PATH) as connection:
        orders = pd.read_sql_query(
            """
            SELECT order_id, customer_id, order_amount
            FROM orders
            WHERE LOWER(status) = 'completed'
            """,
            connection,
        )

    # Calculate revenue independently using pandas
    total_revenue = round(orders["order_amount"].sum(), 2)

    print("Completed orders:", len(orders))
    print("Pandas total revenue:", total_revenue)

    expected_revenue = 2998668.80

    assert len(orders) == 11110, "Unexpected completed-order count"
    assert abs(total_revenue - expected_revenue) < 0.01, (
        "Revenue does not match the expected value"
    )

    print("PANDAS REVENUE CHECK PASSED")


if __name__ == "__main__":
    main()
