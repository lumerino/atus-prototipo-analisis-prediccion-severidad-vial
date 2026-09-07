from __future__ import annotations

import streamlit as st

from .utils import query


def render(_: dict[str, object]) -> None:
    st.subheader("Tendencia anual 1997-2024")
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
    st.line_chart(annual.set_index("anio")[["accidentes_validos", "accidentes_con_victimas"]])
    st.subheader("Proporción de accidentes con víctimas")
    st.area_chart(annual.set_index("anio")[["proporcion_con_victimas"]])
    st.dataframe(annual, width="stretch", hide_index=True)
