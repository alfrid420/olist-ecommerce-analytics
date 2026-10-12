"""Load the cleaned orders into SQLite and (optionally) run the queries.

Usage:
    python src/load_sql.py                 # build data/olist.db
    python src/load_sql.py --run-queries   # also run src/queries.sql
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "clean_orders.csv"
DB_PATH = ROOT / "data" / "olist.db"
SQL_PATH = Path(__file__).resolve().parent / "queries.sql"
TABLE = "orders"

# Columns that queries filter or group on frequently.
INDEXED_COLUMNS = ("order_month", "customer_id", "customer_state", "product_category")


def load_orders(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> int:
    """Write the CSV into the ``orders`` table and return the row count."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python src/clean_data.py` first."
        )
    df = pd.read_csv(csv_path)
    # SQLite has no boolean type: store the flag as 0/1 integers.
    df["is_repeat_customer"] = df["is_repeat_customer"].astype(int)
    df["delivery_date_imputed"] = df["delivery_date_imputed"].astype(int)

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.begin() as conn:  # one transaction: all or nothing
            df.to_sql(TABLE, conn, if_exists="replace", index=False, chunksize=10_000)
            for col in INDEXED_COLUMNS:
                conn.execute(
                    text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_{col} ON {TABLE}({col})")
                )
            count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE}")).scalar_one()
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Database load failed: {exc}") from exc
    finally:
        engine.dispose()

    if count != len(df):
        raise RuntimeError(f"Row mismatch: CSV has {len(df)}, table has {count}")
    return int(count)


def split_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split the SQL file into (title, statement) pairs.

    Titles come from the ``-- Qn: ...`` comment above each statement.
    """
    queries = []
    for chunk in sql_text.split(";"):
        # Skip chunks that contain only comments / whitespace.
        body_lines = [
            ln for ln in chunk.splitlines() if ln.strip() and not ln.strip().startswith("--")
        ]
        if not body_lines:
            continue
        match = re.search(r"^--\s*(Q\d+:.*)$", chunk, flags=re.MULTILINE)
        title = match.group(1).strip() if match else "Query"
        queries.append((title, chunk.strip()))
    return queries


def run_queries(db_path: Path = DB_PATH, sql_path: Path = SQL_PATH) -> None:
    """Execute every query in ``queries.sql`` and print the first rows."""
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            for title, statement in split_queries(sql_path.read_text()):
                print(f"\n{title}\n" + "-" * len(title))
                result = pd.read_sql(text(statement), conn)
                print(result.head(12).to_string(index=False))
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Query failed: {exc}") from exc
    finally:
        engine.dispose()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-queries", action="store_true", help="run queries.sql after loading"
    )
    args = parser.parse_args()
    try:
        count = load_orders()
        print(f"Loaded {count:,} rows into table '{TABLE}' -> {DB_PATH}")
        if args.run_queries:
            run_queries()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
