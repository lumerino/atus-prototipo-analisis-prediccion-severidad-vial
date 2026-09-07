from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .estilo import AZUL, NARANJA, section_title
from .utils import query


def render(_: dict[str, object]) -> None:
    section_title("🔮", "Pronóstico agregado")
    st.caption("Fallback lineal local sobre la serie histórica poblacional. Es una aproximación exploratoria, no un modelo Prophet.")

    annual = query(
        """
        SELECT CAST(anio AS INTEGER) AS anio,
               CAST(accidentes_con_victimas AS INTEGER) AS con_victimas
        FROM resumen_anual
        ORDER BY anio
        """
    )
    x = annual["anio"].to_numpy()
    y = annual["con_victimas"].to_numpy()
    coef = np.polyfit(x, y, deg=1)
    future_years = np.arange(x.max() + 1, x.max() + 6)
    forecast = pd.DataFrame(
        {
            "anio": future_years,
            "valor": np.maximum(np.polyval(coef, future_years), 0).round().astype(int),
        }
    )

    historico = annual.rename(columns={"con_victimas": "valor"}).assign(serie="Histórico")
    forecast_serie = forecast.assign(serie="Pronóstico")
    # une el ultimo punto historico al primero del pronostico para que la linea no se corte
    puente = historico.tail(1).assign(serie="Pronóstico")
    combinado = pd.concat([historico, puente, forecast_serie], ignore_index=True)

    chart = (
        alt.Chart(combinado)
        .mark_line(point=alt.OverlayMarkDef(size=35), strokeWidth=2.5)
        .encode(
            x=alt.X("anio:O", title="Año"),
            y=alt.Y("valor:Q", title="Accidentes con víctimas", axis=alt.Axis(format=",.0f")),
            color=alt.Color(
                "serie:N",
                title="Serie",
                scale=alt.Scale(domain=["Histórico", "Pronóstico"], range=[AZUL, NARANJA]),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.StrokeDash(
                "serie:N",
                scale=alt.Scale(domain=["Histórico", "Pronóstico"], range=[[1, 0], [6, 3]]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[alt.Tooltip("anio:O", title="Año"), "serie:N", alt.Tooltip("valor:Q", title="Accidentes", format=",.0f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("**Próximos 5 años proyectados**")
    cols = st.columns(len(forecast))
    for col, (_, row) in zip(cols, forecast.iterrows()):
        col.metric(str(int(row["anio"])), f"{int(row['valor']):,}")
