from __future__ import annotations

import altair as alt
import streamlit as st

from .estilo import AZUL, SECUENCIAL_AZUL, section_title
from .utils import filter_clause, query


def render(filters: dict[str, object]) -> None:
    where, params = filter_clause(filters)
    section_title("🚦", "Perfil de severidad")
    st.caption("Proporción de accidentes con víctimas por tipo de accidente, hora del día y variables del conductor.")

    by_type = query(
        f"""
        SELECT TIPACCID AS tipo_accidente,
               COUNT(*) AS accidentes,
               AVG(CAST(severidad_binaria AS INTEGER)) AS proporcion_con_victimas
        FROM hechos_accidentes
        {where}
        GROUP BY TIPACCID
        ORDER BY accidentes DESC
        LIMIT 15
        """,
        tuple(params),
    )
    by_hour = query(
        f"""
        SELECT CAST(ID_HORA AS INTEGER) AS hora,
               COUNT(*) AS accidentes,
               AVG(CAST(severidad_binaria AS INTEGER)) AS proporcion_con_victimas
        FROM hechos_accidentes
        {where}
        GROUP BY CAST(ID_HORA AS INTEGER)
        ORDER BY hora
        """,
        tuple(params),
    )
    # ID_HORA=99 codifica "hora no especificada" (misma convencion que el modelo,
    # ver franja_from_hora en generar_modelado_atus.py); se excluye de la
    # tendencia horaria porque no es una hora real del dia.
    by_hour = by_hour[(by_hour["hora"] >= 0) & (by_hour["hora"] <= 23)]
    by_driver = query(
        f"""
        SELECT SEXO AS sexo,
               ALIENTO AS aliento,
               CINTURON AS cinturon,
               COUNT(*) AS accidentes,
               AVG(CAST(severidad_binaria AS INTEGER)) AS proporcion_con_victimas
        FROM hechos_accidentes
        {where}
        GROUP BY SEXO, ALIENTO, CINTURON
        ORDER BY accidentes DESC
        LIMIT 20
        """,
        tuple(params),
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**% con víctimas por tipo de accidente**")
        chart_type = (
            alt.Chart(by_type)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("proporcion_con_victimas:Q", title="% con víctimas", axis=alt.Axis(format=".0%")),
                y=alt.Y("tipo_accidente:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                color=alt.Color(
                    "proporcion_con_victimas:Q",
                    scale=alt.Scale(range=SECUENCIAL_AZUL),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("tipo_accidente:N", title="Tipo de accidente"),
                    alt.Tooltip("accidentes:Q", title="Accidentes", format=",.0f"),
                    alt.Tooltip("proporcion_con_victimas:Q", title="% con víctimas", format=".1%"),
                ],
            )
            .properties(height=380)
        )
        st.altair_chart(chart_type, use_container_width=True)

    with c2:
        st.markdown("**% con víctimas por hora del día**")
        chart_hour = (
            alt.Chart(by_hour)
            .mark_line(point=alt.OverlayMarkDef(size=40, color=AZUL), color=AZUL, strokeWidth=2.5)
            .encode(
                x=alt.X("hora:O", title="Hora"),
                y=alt.Y("proporcion_con_victimas:Q", title="% con víctimas", axis=alt.Axis(format=".0%")),
                tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("proporcion_con_victimas:Q", title="% con víctimas", format=".1%")],
            )
            .properties(height=380)
        )
        st.altair_chart(chart_hour, use_container_width=True)

    st.markdown("**Combinaciones de variables del conductor**")
    by_driver_mostrar = by_driver.assign(proporcion_con_victimas=(by_driver["proporcion_con_victimas"] * 100).round(1))
    st.dataframe(
        by_driver_mostrar,
        width="stretch",
        hide_index=True,
        column_config={
            "proporcion_con_victimas": st.column_config.ProgressColumn(
                "% con víctimas", format="%.1f%%", min_value=0.0, max_value=100.0
            ),
        },
    )
