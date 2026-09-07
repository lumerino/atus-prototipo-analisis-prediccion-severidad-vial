from __future__ import annotations

import calendar
import json
import sys
from datetime import date
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from .estilo import AZUL, SECUENCIAL_AZUL, probabilidad_color, section_title
from .utils import ROOT, query


MODEL_PATH = ROOT / "modelos" / "modelo_severidad_histgb.joblib"
METADATA_PATH = ROOT / "modelos" / "metadata_modelo.json"
# El pipeline persistido referencia `to_dense` como `generar_modelado_atus.to_dense`
# al des-serializarse con joblib; ese modulo vive en src/ (vendorizado, ver
# generar_modelado_atus.py), asi que basta con que src/ este en sys.path.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ETIQUETAS_CAMPO = {
    "TIPACCID": "Tipo de accidente",
    "CAUSAACCI": "Causa del accidente",
    "CAPAROD": "Superficie de rodamiento",
    "SEXO": "Sexo del conductor",
    "ALIENTO": "Aliento alcohólico",
    "CINTURON": "Uso de cinturón",
}

# URBANA y SUBURBANA son, en los datos crudos, dos caras de una misma variable
# de zona (un accidente urbano siempre trae SUBURBANA="Sin accidente en esta
# zona" y viceversa). Exponerlas como dos selectbox independientes permite
# combinaciones contradictorias (p. ej. "urbana en intersección" + "suburbana
# en camino rural" a la vez). Se muestran como una sola opción de zona.
ZONA_OPCIONES = {
    "Urbana — accidente en intersección": ("Accidente en intersección", "Sin accidente en esta zona"),
    "Urbana — accidente fuera de intersección": ("Accidente en no intersección", "Sin accidente en esta zona"),
    "Suburbana — camino rural": ("Sin accidente en esta zona", "Accidente en camino rural"),
    "Suburbana — carretera estatal": ("Sin accidente en esta zona", "Accidente en carretera estatal"),
    "Suburbana — otro camino": ("Sin accidente en esta zona", "Accidentes en otro camino"),
}


def _franja_horaria(hora: int) -> str:
    """Misma regla que `franja_from_hora` en generar_modelado_atus.py, para que
    la franja horaria del simulador sea siempre consistente con la Hora elegida
    en vez de un campo independiente que el usuario podria dejar contradictorio."""
    if 0 <= hora <= 5:
        return "Madrugada"
    if 6 <= hora <= 11:
        return "Mañana"
    if 12 <= hora <= 17:
        return "Tarde"
    if 18 <= hora <= 23:
        return "Noche"
    return "No especificada"


NOMBRES_MES = {
    "01": "Enero", "02": "Febrero", "03": "Marzo", "04": "Abril",
    "05": "Mayo", "06": "Junio", "07": "Julio", "08": "Agosto",
    "09": "Septiembre", "10": "Octubre", "11": "Noviembre", "12": "Diciembre",
}
_DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def _dia_semana(anio: int, mes: int, dia: int) -> tuple[str, int]:
    """Dado Año+Mes+Día del mes, calcula el día de la semana real (calendario
    Gregoriano) en vez de dejarlo como un campo independiente: en un calendario
    real el día de la semana queda determinado por la fecha, no es una
    variable que se pueda elegir aparte sin arriesgar una combinación
    imposible (p. ej. "15 de enero de 2024" que en realidad fue lunes, con
    "día de la semana = domingo" elegido a mano). Si el día no existe en ese
    mes/año (31 de febrero), se recorta al último día válido del mes."""
    dias_en_mes = calendar.monthrange(anio, mes)[1]
    dia_ajustado = min(dia, dias_en_mes)
    fecha = date(anio, mes, dia_ajustado)
    return _DIAS_SEMANA[fecha.weekday()], dia_ajustado

METRICA_ETIQUETAS = {
    "accuracy": ("Accuracy", "Exactitud global"),
    "precision": ("Precisión", "De lo predicho positivo"),
    "recall": ("Recall", "Cobertura de víctimas reales"),
    "f1": ("F1", "Balance precisión/recall"),
    "roc_auc": ("ROC-AUC", "Discriminación del modelo"),
}


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _matriz_confusion_chart(cm: list[list[int]]) -> alt.Chart:
    etiquetas = ["Sólo daños", "Con víctimas"]
    total = sum(sum(row) for row in cm)
    filas = [
        {"real": etiquetas[i], "prediccion": etiquetas[j], "conteo": cm[i][j], "pct": cm[i][j] / total}
        for i in range(2)
        for j in range(2)
    ]
    df = pd.DataFrame(filas)
    base = alt.Chart(df).encode(
        x=alt.X("prediccion:N", title="Predicción", sort=etiquetas),
        y=alt.Y("real:N", title="Real", sort=etiquetas),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("conteo:Q", scale=alt.Scale(range=SECUENCIAL_AZUL), legend=None),
        tooltip=[alt.Tooltip("real:N"), alt.Tooltip("prediccion:N"), alt.Tooltip("conteo:Q", format=",.0f"), alt.Tooltip("pct:Q", format=".1%")],
    )
    texto = base.mark_text(fontWeight="bold", fontSize=15).encode(
        text=alt.Text("conteo:Q", format=",.0f"),
        color=alt.condition(alt.datum.pct > 0.15, alt.value("white"), alt.value("#0b0b0b")),
    )
    return (heat + texto).properties(height=260)


def _roc_chart(roc: pd.DataFrame) -> alt.Chart:
    modelo = alt.Chart(roc).mark_line(color=AZUL, strokeWidth=2.5).encode(
        x=alt.X("fpr:Q", title="Tasa de falsos positivos", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("tpr:Q", title="Tasa de verdaderos positivos", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("fpr:Q", format=".2f"), alt.Tooltip("tpr:Q", format=".2f")],
    )
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color="#c3c2b7", strokeDash=[5, 4]
    ).encode(x="x:Q", y="y:Q")
    return (modelo + diagonal).properties(height=260)


def render(_: dict[str, object]) -> None:
    section_title("🤖", "Modelo predictivo")

    if not MODEL_PATH.exists():
        st.error("No se encontró el modelo. Ejecuta: python src/entrenar_modelo.py")
        st.stop()

    metadata = load_metadata()
    metrics = metadata["metricas_prueba_2024"]

    st.markdown("**Desempeño en prueba temporal 2024** (censo completo, no visto durante el entrenamiento)")
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        etiqueta, ayuda = METRICA_ETIQUETAS[key]
        col.metric(etiqueta, f"{metrics[key]:.3f}", help=ayuda)

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("**Matriz de confusión**")
        st.altair_chart(_matriz_confusion_chart(metadata["matriz_confusion"]), use_container_width=True)
    with right:
        st.markdown(f"**Curva ROC** (AUC = {metrics['roc_auc']:.3f})")
        roc = pd.DataFrame(metadata["roc_curve"])
        st.altair_chart(_roc_chart(roc), use_container_width=True)

    st.markdown("---")
    section_title("🎯", "Simulador de probabilidad")
    st.caption("Arma un escenario hipotético y el pipeline persistido calcula la probabilidad de que el accidente tenga víctimas.")

    pipeline = load_model()
    cat_values = metadata["categorical_values"]
    defaults = metadata["numeric_defaults"]
    entidades = query("SELECT ID_ENTIDAD, NOM_ENTIDAD FROM dim_entidad ORDER BY ID_ENTIDAD")

    with st.form("simulador"):
        st.markdown("##### 🕒 Cuándo y quién")
        st.caption("El día de la semana se calcula solo a partir del año, mes y día del mes (no se elige aparte).")
        c1, c2, c3 = st.columns(3)
        row = {key: defaults[key] for key in metadata["numeric_features"]}
        row["ANIO"] = c1.number_input("Año", min_value=1997, max_value=2026, value=2024)
        mes_opciones = cat_values.get("MES", [f"{m:02d}" for m in range(1, 13)])
        row["MES"] = c2.selectbox("Mes", mes_opciones, index=0, format_func=lambda m: NOMBRES_MES.get(m, m))
        row["ID_DIA"] = c3.number_input("Día del mes", min_value=1, max_value=31, value=15)
        row["ID_HORA"] = c1.number_input("Hora", min_value=0, max_value=23, value=18)
        row["edad_valida"] = c2.number_input("Edad conductor", min_value=1, max_value=98, value=35)

        row["DIASEMANA"], dia_ajustado = _dia_semana(int(row["ANIO"]), int(row["MES"]), int(row["ID_DIA"]))
        if dia_ajustado != row["ID_DIA"]:
            st.caption(
                f"⚠️ {NOMBRES_MES.get(row['MES'], row['MES'])} {int(row['ANIO'])} no tiene el día {int(row['ID_DIA'])}; "
                f"se ajustó al día {dia_ajustado}."
            )
            row["ID_DIA"] = dia_ajustado
        st.caption(
            f"📅 Fecha: {int(row['ID_DIA']):02d}/{row['MES']}/{int(row['ANIO'])} → día de la semana derivado: "
            f"**{row['DIASEMANA']}**."
        )
        row["edad_no_especificada"] = 0

        st.markdown("##### 🚗 Vehículos involucrados")
        st.caption("El total de vehículos se calcula automáticamente sumando los siguientes campos.")
        v1, v2, v3, v4 = st.columns(4)
        row["AUTOMOVIL"] = v1.number_input("Automóviles", min_value=0, max_value=20, value=1)
        row["MOTOCICLET"] = v2.number_input("Motocicletas", min_value=0, max_value=20, value=1)
        row["BICICLETA"] = v3.number_input("Bicicletas", min_value=0, max_value=20, value=0)
        row["CAMIONETA"] = v4.number_input("Camionetas", min_value=0, max_value=20, value=0)
        row["total_vehiculos"] = max(1, row["AUTOMOVIL"] + row["MOTOCICLET"] + row["BICICLETA"] + row["CAMIONETA"])
        row["involucra_motocicleta"] = int(row["MOTOCICLET"] > 0)
        row["involucra_bicicleta"] = int(row["BICICLETA"] > 0)
        row["involucra_pesado"] = 0

        st.markdown("##### 📍 Lugar y contexto")
        preferred_defaults = {
            "CAUSAACCI": "Conductor",
            "CAPAROD": "Pavimentada",
            "SEXO": "Hombre",
            "ALIENTO": "No",
            "CINTURON": "Sí",
            "TIPACCID": "Colisión con vehículo automotor",
        }
        g1, g2, g3 = st.columns(3)
        grupos = [g1, g2, g3]
        entidad_label = g1.selectbox(
            "Entidad",
            [f"{r.ID_ENTIDAD} - {r.NOM_ENTIDAD}" for r in entidades.itertuples(index=False)],
            index=18,
        )
        row["ID_ENTIDAD"] = entidad_label.split(" - ", 1)[0]

        zona_label = g2.selectbox("Zona y tipo de vía", list(ZONA_OPCIONES.keys()), index=0)
        row["URBANA"], row["SUBURBANA"] = ZONA_OPCIONES[zona_label]

        excluidos = {"ID_ENTIDAD", "URBANA", "SUBURBANA", "franja_horaria", "MES", "DIASEMANA"}
        otras_categoricas = [f for f in metadata["categorical_features"] if f not in excluidos]
        for i, feature in enumerate(otras_categoricas):
            options = cat_values.get(feature, ["No especificado"])
            preferred = preferred_defaults.get(feature)
            default_index = options.index(preferred) if preferred in options else 0
            etiqueta = ETIQUETAS_CAMPO.get(feature, feature)
            row[feature] = grupos[(i + 2) % 3].selectbox(etiqueta, options, index=default_index)

        row["franja_horaria"] = _franja_horaria(int(row["ID_HORA"]))
        st.caption(f"Franja horaria derivada de la hora ({row['ID_HORA']:02d}:00): **{row['franja_horaria']}**.")

        submitted = st.form_submit_button("🧮 Calcular probabilidad", use_container_width=True)

    if submitted:
        frame = pd.DataFrame([row])[metadata["numeric_features"] + metadata["categorical_features"]]
        probability = float(pipeline.predict_proba(frame)[0, 1])
        color, etiqueta_riesgo = probabilidad_color(probability)

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(
                f"""
                <div class="atus-card" style="text-align:center;">
                    <div style="color:#52514e; font-weight:600; font-size:0.9rem;">Probabilidad estimada</div>
                    <div style="font-size:2.4rem; font-weight:800; color:{color};">{probability:.1%}</div>
                    <span class="atus-badge" style="background:{color};">{etiqueta_riesgo}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            gauge = (
                alt.Chart(pd.DataFrame({"valor": [probability], "resto": [1 - probability]}))
                .transform_fold(["valor", "resto"])
                .mark_arc(innerRadius=55, outerRadius=85)
                .encode(
                    theta=alt.Theta("value:Q", stack=True),
                    color=alt.Color(
                        "key:N",
                        scale=alt.Scale(domain=["valor", "resto"], range=[color, "#e1e0d9"]),
                        legend=None,
                    ),
                )
                .properties(height=180, width=180)
            )
            st.altair_chart(gauge, use_container_width=False)
            st.caption(
                "Umbrales orientativos: <30% bajo, 30-55% medio, 55-75% alto, >75% muy alto. "
                "Es apoyo para priorización, no una decisión automática (ver sección 14.1 del documento)."
            )
