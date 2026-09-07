from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .utils import query


def render(_: dict[str, object]) -> None:
    st.subheader("Pronostico agregado simplificado")
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
            "pronostico_con_victimas": np.maximum(np.polyval(coef, future_years), 0).round().astype(int),
        }
    )
    combined = pd.concat(
        [
            annual.rename(columns={"con_victimas": "historico"}).assign(pronostico_con_victimas=np.nan),
            forecast.assign(historico=np.nan),
        ],
        ignore_index=True,
    )
    st.caption("Fallback lineal local. Es una aproximacion exploratoria, no un modelo Prophet.")
    st.line_chart(combined.set_index("anio")[["historico", "pronostico_con_victimas"]])
    st.dataframe(forecast, width="stretch", hide_index=True)
