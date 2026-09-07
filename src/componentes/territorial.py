from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from .estilo import AZUL, section_title
from .utils import query


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _bar(frame: pd.DataFrame, campo: str, titulo_eje: str) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_bar(color=AZUL, cornerRadiusEnd=4)
        .encode(
            x=alt.X("accidentes_totales:Q", title="Accidentes totales", axis=alt.Axis(format=",.0f")),
            y=alt.Y(f"{campo}:N", sort="-x", title=titulo_eje, axis=alt.Axis(labelLimit=220)),
            tooltip=[
                alt.Tooltip(f"{campo}:N", title=titulo_eje),
                alt.Tooltip("accidentes_totales:Q", title="Accidentes", format=",.0f"),
                alt.Tooltip("porcentaje_total:Q", title="% del total", format=".2f"),
            ],
        )
        .properties(height=340)
    )


def render(_: dict[str, object]) -> None:
    section_title("🗺️", "Distribución territorial")
    st.caption("Top 10 entidades y municipios por volumen total de accidentes (serie histórica del EDA).")

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
    with left:
        st.markdown("**Top 10 entidades**")
        st.altair_chart(_bar(entidades, "entidad", "Entidad"), use_container_width=True)
    with right:
        st.markdown("**Top 10 municipios**")
        st.altair_chart(_bar(municipios, "municipio", "Municipio"), use_container_width=True)

    with st.expander("Ver tablas de datos"):
        c1, c2 = st.columns(2)
        c1.dataframe(entidades, width="stretch", hide_index=True)
        c2.dataframe(municipios, width="stretch", hide_index=True)
