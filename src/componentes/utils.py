from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "datos" / "atus_prototipo.db"


@st.cache_data(show_spinner=False)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def require_db() -> None:
    if not DB_PATH.exists():
        st.error("No se encontró la base SQLite. Ejecuta: python src/construir_base_datos.py")
        st.stop()


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def filter_clause(filters: dict[str, object]) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if filters.get("anio") != "Todos":
        clauses.append("ANIO = ?")
        params.append(str(filters["anio"]))
    if filters.get("entidad") != "Todas":
        clauses.append("ID_ENTIDAD = ?")
        params.append(str(filters["entidad"]).zfill(2))
    if filters.get("tipo") != "Todos":
        clauses.append("TIPACCID = ?")
        params.append(filters["tipo"])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params
