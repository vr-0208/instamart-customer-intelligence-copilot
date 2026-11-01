
from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"
SQL_PATH = PROJECT_ROOT / "sql" / "queries.sql"


def main():
    # Read all seven SQL queries from the file
    sql_text = SQL_PATH.read_text(encoding="utf-8")

    # Split the file into individual queries
    queries = [
        query.strip()
        for query in sql_text.split(";")
        if query.strip()
    ]

    print(f"Found {len(queries)} SQL queries.")

    # Connect to the database and execute each query
    with sqlite3.connect(DB_PATH) as connection:
        for index, query in enumerate(queries, start=1):
            result = pd.read_sql_query(query, connection)

            print(f"\n{'=' * 50}")
            print(f"QUERY {index}")
            print(f"{'=' * 50}")
            print("Rows returned:", len(result))
            print(result.head(5).to_string(index=False))
            if index == 6:
                print(
                    "Rows where RANK differs from ROW_NUMBER():",
                    (result["spending_rank"] != result["row_num"]).sum()
                )

            if index == 2:
                print(
                    "Total revenue:",
                    round(result["total_revenue"].sum(), 2)
                )

            if index == 7:
                if not result.empty:
                    print(
                        "Final cumulative revenue:",
                        result["cumulative_revenue"].iloc[-1]
                    )


if __name__ == "__main__":
    main()
