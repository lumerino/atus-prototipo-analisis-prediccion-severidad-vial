from __future__ import annotations

import streamlit as st

from componentes import modelo_predictivo, pronostico, resumen, severidad, tendencias, territorial
from componentes.utils import query, require_db


st.set_page_config(page_title="ATUS | Prototipo de severidad vial", layout="wide")
require_db()

st.title("ATUS | Prototipo de analisis y prediccion de severidad vial")
st.caption("Base ATUS INEGI 1997-2024. Prototipo local del Entregable 4.")

with st.sidebar:
    st.header("Filtros")
    years = ["Todos"] + [str(x) for x in query("SELECT DISTINCT ANIO FROM hechos_accidentes ORDER BY ANIO")["ANIO"]]
    entities = query("SELECT ID_ENTIDAD, NOM_ENTIDAD FROM dim_entidad ORDER BY ID_ENTIDAD")
    entity_options = ["Todas"] + [f"{r.ID_ENTIDAD} - {r.NOM_ENTIDAD}" for r in entities.itertuples(index=False)]
    types = ["Todos"] + query("SELECT DISTINCT TIPACCID FROM hechos_accidentes ORDER BY TIPACCID")["TIPACCID"].astype(str).tolist()

    section = st.radio(
        "Seccion",
        [
            "Resumen general",
            "Tendencias",
            "Distribucion territorial",
            "Perfil de severidad",
            "Modelo predictivo",
            "Pronostico agregado",
        ],
    )
    selected_year = st.selectbox("Ano", years)
    selected_entity = st.selectbox("Entidad", entity_options)
    selected_type = st.selectbox("Tipo de accidente", types)

filters = {
    "anio": selected_year,
    "entidad": selected_entity.split(" - ", 1)[0] if selected_entity != "Todas" else "Todas",
    "tipo": selected_type,
}

if section == "Resumen general":
    resumen.render(filters)
elif section == "Tendencias":
    tendencias.render(filters)
elif section == "Distribucion territorial":
    territorial.render(filters)
elif section == "Perfil de severidad":
    severidad.render(filters)
elif section == "Modelo predictivo":
    modelo_predictivo.render(filters)
else:
    pronostico.render(filters)
