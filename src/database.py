import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = (
    PROJECT_ROOT
    /"database"
    /"supplier.db"
)

def get_connection():
    """
    Return a connection to the Supplier Intelligence SQLite Database
    """

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return sqlite3.connect(DB_PATH)
