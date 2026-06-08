"""
etl.py
======
Extract-Transform-Load layer for the Credit Risk Predictor.

PRIMARY PATH (default): CSV  ->  MySQL  ->  pandas DataFrame.
The MySQL store is the single source of truth for analysis, mirroring how a
risk team would query a warehouse rather than read flat files.

FALLBACK PATH: read straight from the CSV. This keeps the repository runnable
on a machine without a database (e.g. a recruiter cloning the repo, or CI),
without changing the modelling code. Select it with `source="csv"`.

Database credentials are read from environment variables so that no secrets
live in source control:

    export CREDIT_DB_USER=root
    export CREDIT_DB_PASSWORD=yourpassword
    export CREDIT_DB_HOST=localhost
    export CREDIT_DB_NAME=Credit_Risk_Project
"""

from __future__ import annotations

import os
import pandas as pd


def _db_config() -> dict:
    """Read DB connection settings from the environment (with sane defaults)."""
    return {
        "user": os.environ.get("CREDIT_DB_USER", "root"),
        "password": os.environ.get("CREDIT_DB_PASSWORD", ""),
        "host": os.environ.get("CREDIT_DB_HOST", "localhost"),
        "database": os.environ.get("CREDIT_DB_NAME", "Credit_Risk_Project"),
    }


def _engine():
    """Build a SQLAlchemy engine. Imported lazily so the CSV path needs no DB driver."""
    from sqlalchemy import create_engine

    cfg = _db_config()
    url = (
        f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}/{cfg['database']}"
    )
    return create_engine(url)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw frame: drop the unnamed index column, tidy column
    names, and fill missing account categories with an explicit 'unknown'
    label (missingness is itself informative for credit risk)."""
    df = df.copy()
    # The source CSV ships with an unnamed integer index as the first column.
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed") or c == ""]
    df = df.drop(columns=unnamed, errors="ignore")
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")
    # Treat literal "NA" strings and real NaNs alike.
    df = df.replace("NA", pd.NA).fillna("unknown")
    return df


def load_csv_to_mysql(csv_path: str = "german_credit_data.csv",
                      table: str = "loans_raw") -> int:
    """ETL step: read the CSV, clean it, and (re)load it into MySQL.

    Returns the number of rows written. Requires a reachable MySQL server.
    """
    df = _clean(pd.read_csv(csv_path))
    engine = _engine()
    df.to_sql(table, con=engine, if_exists="replace", index=False)
    return len(df)


def fetch_data(source: str = "mysql",
               csv_path: str = "german_credit_data.csv",
               table: str = "loans_raw") -> pd.DataFrame:
    """Return the analysis-ready DataFrame.

    Parameters
    ----------
    source : {"mysql", "csv"}
        "mysql" (default/primary) reads from the warehouse table; if the table
        is missing it is auto-populated from the CSV first. "csv" bypasses the
        database entirely.
    """
    if source == "csv":
        return _clean(pd.read_csv(csv_path))

    if source == "mysql":
        engine = _engine()
        try:
            return pd.read_sql(f"SELECT * FROM {table}", engine)
        except Exception:
            # Table not present yet -> run the load, then read back.
            load_csv_to_mysql(csv_path, table)
            return pd.read_sql(f"SELECT * FROM {table}", engine)

    raise ValueError(f"Unknown source '{source}'. Use 'mysql' or 'csv'.")


if __name__ == "__main__":
    rows = load_csv_to_mysql()
    print(f"Loaded {rows} rows into MySQL table 'loans_raw'.")
