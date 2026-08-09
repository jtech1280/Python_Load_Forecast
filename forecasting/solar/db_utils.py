#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database Connection Utilities
=============================

Shared helper functions for creating and using database connections with SQLAlchemy.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def build_conn_str(
    driver: str,
    server: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """
    Build a SQLAlchemy connection string for SQL Server.
    """
    driver_safe = quote_plus(driver)
    if username and password:
        return f"mssql+pyodbc://{username}:{quote_plus(password)}@{server}/{database}?driver={driver_safe}"
    else:
        return f"mssql+pyodbc://{server}/{database}?driver={driver_safe}&trusted_connection=yes"


def connect(
    driver: str,
    server: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Engine:
    """
    Create a SQLAlchemy engine for SQL Server.
    """
    conn_str = build_conn_str(
        driver=driver,
        server=server,
        database=database,
        username=username,
        password=password,
    )
    logging.info("Connecting to SERVER=%s, DATABASE=%s", server, database)
    engine = create_engine(conn_str)
    # Test the connection
    try:
        connection = engine.connect()
        connection.close()
        logging.info("Connection successful.")
    except Exception as e:
        logging.error("Connection failed: %s", e)
        raise
    return engine


def read_sql(
    engine: Engine, sql: str, params: Optional[Iterable] = None
) -> pd.DataFrame:
    """
    Read a SQL query into pandas using a SQLAlchemy engine.
    """
    return pd.read_sql_query(sql, engine, params=params)
