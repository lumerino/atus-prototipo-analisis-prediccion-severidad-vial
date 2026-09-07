from __future__ import annotations

import logging

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .estilo import AGUA, AZUL, NARANJA, section_title
from .utils import query

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

HORIZONTE = 5


def _forecast_lineal(annual: pd.DataFrame) -> pd.DataFrame:
    x = annual["anio"].to_numpy()
    y = annual["con_victimas"].to_numpy()
    coef = np.polyfit(x, y, deg=1)
    future_years = np.arange(x.max() + 1, x.max() + 1 + HORIZONTE)
    return pd.DataFrame(
        {"anio": future_years, "valor": np.maximum(np.polyval(coef, future_years), 0).round().astype(int)}
    )


@st.cache_data(show_spinner="Ajustando Prophet...")
def _forecast_prophet(annual: pd.DataFrame) -> pd.DataFrame | None:
    try:
        from prophet import Prophet
    except ImportError:
        return None

    df = pd.DataFrame(
        {"ds": pd.to_datetime(annual["anio"].astype(str) + "-01-01"), "y": annual["con_victimas"]}
    )
    modelo = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    modelo.fit(df)
    futuro = modelo.make_future_dataframe(periods=HORIZONTE, freq="YS")
    pronostico = modelo.predict(futuro)
    pronostico = pronostico.tail(HORIZONTE)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    pronostico["anio"] = pronostico["ds"].dt.year
    pronostico["valor"] = pronostico["yhat"].clip(lower=0).round().astype(int)
    pronostico["valor_min"] = pronostico["yhat_lower"].clip(lower=0).round().astype(int)
    pronostico["valor_max"] = pronostico["yhat_upper"].clip(lower=0).round().astype(int)
    return pronostico[["anio", "valor", "valor_min", "valor_max"]]


def render(_: dict[str, object]) -> None:
    section_title("🔮", "Pronóstico agregado")
    st.caption(
        "Comparación entre un fallback lineal simple y un modelo Prophet real, ambos ajustados sobre la "
        "misma serie histórica poblacional (accidentes con víctimas por año, 1997-2024)."
    )

    annual = query(
        """
        SELECT CAST(anio AS INTEGER) AS anio,
               CAST(accidentes_con_victimas AS INTEGER) AS con_victimas
        FROM resumen_anual
        ORDER BY anio
        """
    )

    lineal = _forecast_lineal(annual)
    prophet_fc = _forecast_prophet(annual)

    historico = annual.rename(columns={"con_victimas": "valor"}).assign(serie="Histórico")
    lineal_serie = lineal.assign(serie="Pronóstico lineal")
    series = [historico, historico.tail(1).assign(serie="Pronóstico lineal"), lineal_serie]
    dominio = ["Histórico", "Pronóstico lineal"]
    rango_color = [AZUL, NARANJA]
    rango_dash = [[1, 0], [6, 3]]

    if prophet_fc is not None:
        prophet_serie = prophet_fc.rename(columns={"valor": "valor"}).assign(serie="Pronóstico Prophet")
        series += [historico.tail(1).assign(serie="Pronóstico Prophet"), prophet_serie[["anio", "valor", "serie"]]]
        dominio.append("Pronóstico Prophet")
        rango_color.append(AGUA)
        rango_dash.append([2, 2])
    else:
        st.warning("Prophet no está disponible en este entorno; se muestra solo el fallback lineal.", icon="⚠️")

    combinado = pd.concat(series, ignore_index=True)

    capas = [
        alt.Chart(combinado)
        .mark_line(point=alt.OverlayMarkDef(size=35), strokeWidth=2.5)
        .encode(
            x=alt.X("anio:O", title="Año"),
            y=alt.Y("valor:Q", title="Accidentes con víctimas", axis=alt.Axis(format=",.0f")),
            color=alt.Color(
                "serie:N", title="Serie", scale=alt.Scale(domain=dominio, range=rango_color), legend=alt.Legend(orient="top", title=None)
            ),
            strokeDash=alt.StrokeDash(
                "serie:N", scale=alt.Scale(domain=dominio, range=rango_dash), legend=alt.Legend(orient="top", title=None)
            ),
            tooltip=[alt.Tooltip("anio:O", title="Año"), "serie:N", alt.Tooltip("valor:Q", title="Accidentes", format=",.0f")],
        )
    ]

    if prophet_fc is not None:
        banda = pd.DataFrame(prophet_fc)
        capas.insert(
            0,
            alt.Chart(banda)
            .mark_area(opacity=0.15, color=AGUA)
            .encode(
                x=alt.X("anio:O"),
                y=alt.Y("valor_min:Q"),
                y2=alt.Y2("valor_max:Q"),
            ),
        )

    st.altair_chart(alt.layer(*capas).properties(height=360), use_container_width=True)

    st.markdown("**Próximos 5 años proyectados**")
    if prophet_fc is not None:
        comparacion = lineal.merge(prophet_fc, on="anio", suffixes=("_lineal", "_prophet"))
        comparacion["diferencia"] = comparacion["valor_prophet"] - comparacion["valor_lineal"]
        comparacion_mostrar = comparacion[["anio", "valor_lineal", "valor_prophet", "diferencia"]].rename(
            columns={
                "anio": "Año",
                "valor_lineal": "Lineal (fallback)",
                "valor_prophet": "Prophet",
                "diferencia": "Diferencia (Prophet − Lineal)",
            }
        )
        st.dataframe(comparacion_mostrar, width="stretch", hide_index=True)
        promedio_dif = comparacion["diferencia"].mean()
        st.caption(
            f"En promedio, Prophet proyecta {abs(promedio_dif):,.0f} accidentes con víctimas "
            f"{'más' if promedio_dif >= 0 else 'menos'} al año que el fallback lineal para 2025-2029. "
            "La banda sombreada es el intervalo de incertidumbre (80%) de Prophet."
        )
    else:
        cols = st.columns(len(lineal))
        for col, (_, row) in zip(cols, lineal.iterrows()):
            col.metric(str(int(row["anio"])), f"{int(row['valor']):,}")
