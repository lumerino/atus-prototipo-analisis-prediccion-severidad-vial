from __future__ import annotations

import streamlit as st

from .utils import filter_clause, query


def render(filters: dict[str, object]) -> None:
    where, params = filter_clause(filters)
    st.subheader("Perfil de severidad")

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
        SELECT ID_HORA AS hora,
               COUNT(*) AS accidentes,
               AVG(CAST(severidad_binaria AS INTEGER)) AS proporcion_con_victimas
        FROM hechos_accidentes
        {where}
        GROUP BY ID_HORA
        ORDER BY CAST(ID_HORA AS INTEGER)
        """,
        tuple(params),
    )
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
    c1.bar_chart(by_type.set_index("tipo_accidente")[["proporcion_con_victimas"]])
    c2.line_chart(by_hour.set_index("hora")[["proporcion_con_victimas"]])
    st.dataframe(by_driver, width="stretch", hide_index=True)
