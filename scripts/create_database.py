
from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "instamart.db"


def main():
    # Connect to the SQLite database.
    # If the file does not exist, SQLite creates it.
    with sqlite3.connect(DB_PATH) as connection:

        # Load cleaned CSV files into database tables.
        customers = pd.read_csv(CLEAN_DIR / "customers.csv")
        products = pd.read_csv(CLEAN_DIR / "products.csv")
        orders = pd.read_csv(CLEAN_DIR / "orders.csv")

        customers.to_sql(
            "customers", connection, if_exists="replace", index=False
        )
        products.to_sql(
            "products", connection, if_exists="replace", index=False
        )
        orders.to_sql(
            "orders", connection, if_exists="replace", index=False
        )

        # Confirm that the tables were created.
        cursor = connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = cursor.fetchall()

        print("Database created at:", DB_PATH)
        print("Tables created:")
        for table in tables:
            print("-", table[0])


if __name__ == "__main__":
    main()
