from __future__ import annotations

import pandas as pd
import streamlit as st

from .utils import query


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def render(_: dict[str, object]) -> None:
    st.subheader("Top entidades y municipios")
    entidades = query(
        """
        SELECT entidad, accidentes_totales, porcentaje_total
        FROM resumen_territorial
        WHERE tabla_origen = 'tabla_07_top_entidades_total'
        ORDER BY CAST(accidentes_totales AS INTEGER) DESC
        LIMIT 10
        """
    )
    municipios = query(
        """
        SELECT entidad || ' / ' || municipio AS municipio, accidentes_totales, porcentaje_total
        FROM resumen_territorial
        WHERE tabla_origen = 'tabla_09_top_municipios_total'
        ORDER BY CAST(accidentes_totales AS INTEGER) DESC
        LIMIT 10
        """
    )
    entidades = _numeric(entidades, ["accidentes_totales", "porcentaje_total"])
    municipios = _numeric(municipios, ["accidentes_totales", "porcentaje_total"])

    left, right = st.columns(2)
    left.bar_chart(entidades.set_index("entidad")[["accidentes_totales"]])
    right.bar_chart(municipios.set_index("municipio")[["accidentes_totales"]])
    st.dataframe(entidades, width="stretch", hide_index=True)
    st.dataframe(municipios, width="stretch", hide_index=True)
