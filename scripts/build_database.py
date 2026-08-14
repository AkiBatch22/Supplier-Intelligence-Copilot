from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.database import get_connection


DATA_DIR = PROJECT_ROOT / "data"


TABLE_FILES = {
    "suppliers": "suppliers.csv",
    "monthly_performance": "monthly_performance.csv",
    "invoices": "invoices.csv",
    "incidents": "incidents.csv",
    "supplier_reviews": "supplier_reviews.csv",
}


def build_database():

    with get_connection() as conn:

        for table_name, file_name in TABLE_FILES.items():

            file_path = DATA_DIR / file_name

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found: {file_path}"
                )

            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError(
                    f"{file_name} is empty."
                )

            df.to_sql(
                table_name,
                conn,
                if_exists="replace",
                index=False
            )

            print(
                f"{table_name:<25}"
                f"{len(df):>8,} rows"
            )

    print(
        "\nSupplier database created successfully."
    )


if __name__ == "__main__":
    build_database()