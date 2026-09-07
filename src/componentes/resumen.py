from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from .estilo import AZUL, NARANJA, section_title
from .utils import filter_clause, format_pct, query


def render(filters: dict[str, object]) -> None:
    where, params = filter_clause(filters)
    kpis = query(
        f"""
        SELECT
            COUNT(*) AS total,
            SUM(CAST(severidad_binaria AS INTEGER)) AS victimas,
            MIN(CAST(ANIO AS INTEGER)) AS anio_min,
            MAX(CAST(ANIO AS INTEGER)) AS anio_max
        FROM hechos_accidentes
        {where}
        """,
        tuple(params),
    ).iloc[0]
    total = int(kpis["total"] or 0)
    victimas = int(kpis["victimas"] or 0)
    prop = victimas / total if total else 0

    section_title("📊", "Resumen general")
    st.caption("KPIs y tendencia anual del subconjunto de datos seleccionado en los filtros de la izquierda.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accidentes en muestra", f"{total:,}")
    c2.metric("Con víctimas", f"{victimas:,}")
    c3.metric("% con víctimas", format_pct(prop))
    c4.metric("Periodo", f"{int(kpis['anio_min'])}-{int(kpis['anio_max'])}" if total else "Sin datos")

    st.write("")
    st.markdown("**Distribución anual filtrada**")
    annual = query(
        f"""
        SELECT CAST(ANIO AS INTEGER) AS anio,
               COUNT(*) AS accidentes,
               SUM(CAST(severidad_binaria AS INTEGER)) AS con_victimas
        FROM hechos_accidentes
        {where}
        GROUP BY CAST(ANIO AS INTEGER)
        ORDER BY anio
        """,
        tuple(params),
    )
    if annual.empty:
        st.info("No hay registros para los filtros seleccionados.")
        return

    st.caption(
        "⚠️ 2024 es censo completo del año; 1997-2023 es una muestra aleatoria cacheada del "
        "Entregable 2 — los conteos absolutos no son directamente comparables entre ambos "
        "periodos. Para tendencias comparables ver la pestaña **Tendencias**."
    )

    long_df = annual.melt("anio", var_name="serie", value_name="valor")
    long_df["serie"] = long_df["serie"].map({"accidentes": "Accidentes totales", "con_victimas": "Con víctimas"})

    chart = (
        alt.Chart(long_df)
        .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.5)
        .encode(
            x=alt.X("anio:O", title="Año"),
            y=alt.Y("valor:Q", title="Accidentes", axis=alt.Axis(format=",.0f")),
            color=alt.Color(
                "serie:N",
                title=None,
                scale=alt.Scale(domain=["Accidentes totales", "Con víctimas"], range=[AZUL, NARANJA]),
            ),
            tooltip=[alt.Tooltip("anio:O", title="Año"), alt.Tooltip("serie:N", title="Serie"), alt.Tooltip("valor:Q", title="Accidentes", format=",.0f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)
