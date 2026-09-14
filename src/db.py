"""数据层：stdlib sqlite3 + pandas。

刻意不引入 ORM —— 表结构在 sql/schema.sql 里是唯一真相，
换 PostgreSQL 时只需替换本文件的 connect()，上层零改动。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterable, Optional

import pandas as pd

from .config import ROOT, db_path


@contextmanager
def connect():
    conn = sqlite3.connect(db_path())
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema() -> None:
    ddl = (ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(ddl)


def write_df(df: pd.DataFrame, table: str, if_exists: str = "append") -> int:
    if df is None or df.empty:
        return 0
    with connect() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)
    return len(df)


def replace_rows(df: pd.DataFrame, table: str, keys: Iterable[str]) -> int:
    """按主键先删后插，保证重复运行幂等。"""
    if df is None or df.empty:
        return 0
    keys = list(keys)
    with connect() as conn:
        cur = conn.cursor()
        where = " AND ".join(f"{k} = ?" for k in keys)
        cur.executemany(
            f"DELETE FROM {table} WHERE {where}",
            df[keys].itertuples(index=False, name=None),
        )
        df.to_sql(table, conn, if_exists="append", index=False)
    return len(df)


def read_df(sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def table_count(table: str) -> int:
    try:
        return int(read_df(f"SELECT COUNT(*) AS n FROM {table}")["n"].iloc[0])
    except Exception:
        return 0
