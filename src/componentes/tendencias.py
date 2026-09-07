from __future__ import annotations

import altair as alt
import streamlit as st

from .estilo import AZUL, NARANJA, SECUENCIAL_AZUL, section_title
from .utils import query


def render(_: dict[str, object]) -> None:
    section_title("📈", "Tendencias")
    st.caption("Serie histórica poblacional 1997-2024 (censo completo por año, sin muestreo).")

    annual = query(
        """
        SELECT anio,
               accidentes_validos,
               accidentes_con_victimas,
               proporcion_con_victimas
        FROM resumen_anual
        ORDER BY anio
        """
    )

    st.markdown("**Accidentes totales vs. accidentes con víctimas**")
    long_df = annual.melt(
        "anio", value_vars=["accidentes_validos", "accidentes_con_victimas"], var_name="serie", value_name="valor"
    )
    long_df["serie"] = long_df["serie"].map(
        {"accidentes_validos": "Accidentes totales", "accidentes_con_victimas": "Con víctimas"}
    )
    chart1 = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("anio:O", title="Año"),
            y=alt.Y("valor:Q", title="Accidentes", axis=alt.Axis(format=",.0f")),
            color=alt.Color(
                "serie:N",
                title=None,
                scale=alt.Scale(domain=["Accidentes totales", "Con víctimas"], range=[AZUL, NARANJA]),
            ),
            tooltip=[alt.Tooltip("anio:O", title="Año"), "serie:N", alt.Tooltip("valor:Q", format=",.0f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart1, use_container_width=True)

    st.markdown("**Proporción de accidentes con víctimas**")
    chart2 = (
        alt.Chart(annual)
        .mark_area(line={"color": AZUL, "strokeWidth": 2}, interpolate="monotone", opacity=0.85)
        .encode(
            x=alt.X("anio:O", title="Año"),
            y=alt.Y("proporcion_con_victimas:Q", title="Proporción con víctimas", axis=alt.Axis(format=".0%")),
            color=alt.value(SECUENCIAL_AZUL[2]),
            tooltip=[alt.Tooltip("anio:O", title="Año"), alt.Tooltip("proporcion_con_victimas:Q", title="Proporción", format=".1%")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart2, use_container_width=True)

    with st.expander("Ver tabla de datos"):
        st.dataframe(annual, width="stretch", hide_index=True)
