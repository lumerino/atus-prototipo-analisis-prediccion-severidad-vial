from __future__ import annotations

from pathlib import Path

import streamlit as st

from componentes import modelo_predictivo, pronostico, resumen, severidad, tendencias, territorial
from componentes.estilo import inyectar_css, registrar_tema_altair
from componentes.utils import query, require_db

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "unir_logo.png"

st.set_page_config(page_title="ATUS | Prototipo de severidad vial", layout="wide", page_icon="🚦")
inyectar_css()
registrar_tema_altair()
if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large")
require_db()

st.markdown(
    """
    <div class="atus-hero">
        <h1>🚦 ATUS · Prototipo de análisis y predicción de severidad vial</h1>
        <p>Base ATUS INEGI 1997-2024 · Entregable 4 · TFM Máster en Análisis y Visualización de Datos Masivos</p>
    </div>
    """,
    unsafe_allow_html=True,
)

SECCIONES = [
    ("📊", "Resumen general"),
    ("📈", "Tendencias"),
    ("🗺️", "Distribución territorial"),
    ("🚦", "Perfil de severidad"),
    ("🤖", "Modelo predictivo"),
    ("🔮", "Pronóstico agregado"),
]

# Solo estas dos secciones consultan hechos_accidentes (con Año/Entidad/Tipo);
# el resto usa tablas ya agregadas del EDA (resumen_anual, resumen_territorial)
# o metricas fijas del modelo, para no mezclar la muestra 1997-2023 con el
# censo 2024 fuera de donde ya se advierte esa diferencia (ver Resumen general).
SECCIONES_CON_FILTRO = {"Resumen general", "Perfil de severidad"}

with st.sidebar:
    st.markdown(
        """
        <div class="atus-curso">
            <strong>Universidad Internacional de La Rioja (UNIR)</strong><br>
            Escuela Superior de Ingeniería y Tecnología<br>
            Máster Universitario en Análisis y Visualización de Datos Masivos<br>
            Seminario: Innovación en Análisis y Visualización de Datos<br>
            <em>Entregable 4 — Prototipo de la solución</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🧭 Navegación")
    section = st.radio(
        "Sección",
        [nombre for _, nombre in SECCIONES],
        format_func=lambda nombre: f"{dict((n, i) for i, n in SECCIONES)[nombre]}  {nombre}",
        label_visibility="collapsed",
    )

    filtros_aplican = section in SECCIONES_CON_FILTRO

    st.markdown("---")
    st.markdown("### 🔎 Filtros")
    if filtros_aplican:
        st.caption("✅ Estos filtros aplican a esta vista.")
    else:
        st.caption(
            "ℹ️ Estos filtros **no aplican** aquí: esta vista usa series ya "
            "agregadas para todo el país y todo el periodo (ver metodología en "
            "el documento). Solo afectan a *Resumen general* y *Perfil de "
            "severidad*."
        )

    years = ["Todos"] + [str(x) for x in query("SELECT DISTINCT ANIO FROM hechos_accidentes ORDER BY ANIO")["ANIO"]]
    entities = query("SELECT ID_ENTIDAD, NOM_ENTIDAD FROM dim_entidad ORDER BY ID_ENTIDAD")
    entity_options = ["Todas"] + [f"{r.ID_ENTIDAD} - {r.NOM_ENTIDAD}" for r in entities.itertuples(index=False)]
    types = ["Todos"] + query("SELECT DISTINCT TIPACCID FROM hechos_accidentes ORDER BY TIPACCID")["TIPACCID"].astype(str).tolist()

    selected_year = st.selectbox("Año", years, disabled=not filtros_aplican)
    selected_entity = st.selectbox("Entidad", entity_options, disabled=not filtros_aplican)
    selected_type = st.selectbox("Tipo de accidente", types, disabled=not filtros_aplican)

    st.markdown("---")
    st.caption("Prototipo local · Streamlit + SQLite + scikit-learn")

filters = {
    "anio": selected_year,
    "entidad": selected_entity.split(" - ", 1)[0] if selected_entity != "Todas" else "Todas",
    "tipo": selected_type,
}

if section == "Resumen general":
    resumen.render(filters)
elif section == "Tendencias":
    tendencias.render(filters)
elif section == "Distribución territorial":
    territorial.render(filters)
elif section == "Perfil de severidad":
    severidad.render(filters)
elif section == "Modelo predictivo":
    modelo_predictivo.render(filters)
else:
    pronostico.render(filters)
