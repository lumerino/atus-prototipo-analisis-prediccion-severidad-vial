from __future__ import annotations

import streamlit as st

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accidentes en muestra", f"{total:,}")
    c2.metric("Con victimas", f"{victimas:,}")
    c3.metric("% con victimas", format_pct(prop))
    c4.metric("Periodo", f"{int(kpis['anio_min'])}-{int(kpis['anio_max'])}" if total else "Sin datos")

    st.subheader("Distribucion anual filtrada")
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
    else:
        st.line_chart(annual.set_index("anio")[["accidentes", "con_victimas"]])
