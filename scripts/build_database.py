from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.database import DB_PATH


DATA_DIR = PROJECT_ROOT / "data"


TABLE_FILES = {
    "suppliers": "suppliers.csv",
    "monthly_performance": "monthly_performance.csv",
    "invoices": "invoices.csv",
    "incidents": "incidents.csv",
    "supplier_reviews": "supplier_reviews.csv",
}

REQUIRED_COLUMNS = {
    "suppliers": {
        "supplier_id",
        "supplier_name",
        "category",
        "region",
        "criticality",
    },
    "monthly_performance": {
        "supplier_id",
        "month",
        "sla_compliance",
        "on_time_delivery_rate",
        "defect_rate",
        "invoice_accuracy",
        "avg_resolution_days",
        "escalation_count",
    },
    "invoices": {
        "invoice_id",
        "supplier_id",
        "invoice_date",
        "invoice_amount",
        "approved_amount",
        "payment_delay_days",
        "invoice_error_flag",
    },
    "incidents": {
        "incident_id",
        "supplier_id",
        "incident_date",
        "severity",
        "incident_type",
        "description",
        "resolution",
    },
    "supplier_reviews": {
        "review_id",
        "supplier_id",
        "review_date",
        "performance_summary",
        "key_issues",
        "corrective_actions",
        "reviewer_notes",
    },
}


def _load_and_validate_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}

    for table_name, file_name in TABLE_FILES.items():
        file_path = data_dir / file_name
        if not file_path.is_file():
            raise FileNotFoundError(f"Required dataset not found: {file_path}")

        dataframe = pd.read_csv(file_path)
        if dataframe.empty:
            raise ValueError(f"Required dataset is empty: {file_path}")

        missing_columns = REQUIRED_COLUMNS[table_name] - set(dataframe.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{file_name} is missing required columns: {missing}")

        datasets[table_name] = dataframe

    supplier_ids = set(datasets["suppliers"]["supplier_id"])
    if datasets["suppliers"]["supplier_id"].duplicated().any():
        raise ValueError("suppliers.csv contains duplicate supplier_id values")

    for table_name in TABLE_FILES:
        if table_name == "suppliers":
            continue
        unknown_ids = set(datasets[table_name]["supplier_id"]) - supplier_ids
        if unknown_ids:
            examples = ", ".join(sorted(map(str, unknown_ids))[:5])
            raise ValueError(f"{table_name} contains unknown supplier IDs: {examples}")

    return datasets


def _create_indexes(connection: sqlite3.Connection) -> None:
    statements = (
        "CREATE UNIQUE INDEX idx_suppliers_id ON suppliers(supplier_id)",
        "CREATE UNIQUE INDEX idx_suppliers_name ON suppliers(supplier_name)",
        "CREATE INDEX idx_performance_supplier_month ON monthly_performance(supplier_id, month)",
        "CREATE INDEX idx_invoices_supplier_date ON invoices(supplier_id, invoice_date)",
        "CREATE INDEX idx_incidents_supplier_date ON incidents(supplier_id, incident_date)",
        "CREATE INDEX idx_reviews_supplier_date ON supplier_reviews(supplier_id, review_date)",
    )
    for statement in statements:
        connection.execute(statement)


def build_database(
    data_dir: Path = DATA_DIR,
    database_path: Path = DB_PATH,
) -> dict[str, int]:
    """Validate the source CSVs and reproducibly rebuild the SQLite database."""

    data_dir = Path(data_dir)
    database_path = Path(database_path)
    datasets = _load_and_validate_datasets(data_dir)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = tempfile.NamedTemporaryFile(
        prefix="supplier_database_",
        suffix=".tmp",
        dir=database_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    try:
        with closing(sqlite3.connect(temporary_path)) as connection:
            for table_name, dataframe in datasets.items():
                dataframe.to_sql(table_name, connection, if_exists="replace", index=False)
            _create_indexes(connection)
            connection.execute("PRAGMA user_version = 1")
            connection.commit()

        temporary_path.replace(database_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    row_counts = {table: len(dataframe) for table, dataframe in datasets.items()}
    for table_name, row_count in row_counts.items():
        print(f"{table_name:<25}{row_count:>8,} rows")
    print(f"\nSupplier database created successfully: {database_path}")
    return row_counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the supplier SQLite database")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--database", type=Path, default=DB_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    build_database(arguments.data_dir, arguments.database)
