from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from .utils import ROOT, query


MODEL_PATH = ROOT / "modelos" / "modelo_severidad_histgb.joblib"
METADATA_PATH = ROOT / "modelos" / "metadata_modelo.json"
ACTIVIDAD2 = Path(__file__).resolve().parents[4] / "actividad2"
if str(ACTIVIDAD2) not in sys.path:
    sys.path.insert(0, str(ACTIVIDAD2))


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def render(_: dict[str, object]) -> None:
    if not MODEL_PATH.exists():
        st.error("No se encontro el modelo. Ejecuta: python src/entrenar_modelo.py")
        st.stop()

    metadata = load_metadata()
    metrics = metadata["metricas_prueba_2024"]
    st.subheader("Desempeno en prueba temporal 2024")
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(key.upper(), f"{metrics[key]:.3f}")

    left, right = st.columns(2)
    cm = pd.DataFrame(
        metadata["matriz_confusion"],
        index=["Real: solo danos", "Real: con victimas"],
        columns=["Pred: solo danos", "Pred: con victimas"],
    )
    left.dataframe(cm, width="stretch")
    roc = pd.DataFrame(metadata["roc_curve"])
    right.line_chart(roc.set_index("fpr")[["tpr"]])

    st.subheader("Simulador de probabilidad de accidente con victimas")
    pipeline = load_model()
    cat_values = metadata["categorical_values"]
    defaults = metadata["numeric_defaults"]
    entidades = query("SELECT ID_ENTIDAD, NOM_ENTIDAD FROM dim_entidad ORDER BY ID_ENTIDAD")

    with st.form("simulador"):
        c1, c2, c3 = st.columns(3)
        row = {key: defaults[key] for key in metadata["numeric_features"]}
        row["ANIO"] = c1.number_input("Ano", min_value=1997, max_value=2026, value=2024)
        row["ID_DIA"] = c1.number_input("Dia del mes", min_value=1, max_value=31, value=15)
        row["ID_HORA"] = c1.number_input("Hora", min_value=0, max_value=23, value=18)
        row["edad_valida"] = c1.number_input("Edad conductor", min_value=1, max_value=98, value=35)
        row["edad_no_especificada"] = 0
        row["total_vehiculos"] = c2.number_input("Total vehiculos", min_value=1, max_value=20, value=2)
        row["AUTOMOVIL"] = c2.number_input("Automoviles", min_value=0, max_value=20, value=1)
        row["MOTOCICLET"] = c2.number_input("Motocicletas", min_value=0, max_value=20, value=1)
        row["BICICLETA"] = c2.number_input("Bicicletas", min_value=0, max_value=20, value=0)
        row["CAMIONETA"] = c2.number_input("Camionetas", min_value=0, max_value=20, value=0)
        row["involucra_motocicleta"] = int(row["MOTOCICLET"] > 0)
        row["involucra_bicicleta"] = int(row["BICICLETA"] > 0)
        row["involucra_pesado"] = 0

        entidad_label = c3.selectbox(
            "Entidad",
            [f"{r.ID_ENTIDAD} - {r.NOM_ENTIDAD}" for r in entidades.itertuples(index=False)],
            index=18,
        )
        row["ID_ENTIDAD"] = entidad_label.split(" - ", 1)[0]
        for feature in metadata["categorical_features"]:
            if feature == "ID_ENTIDAD":
                continue
            options = cat_values.get(feature, ["No especificado"])
            default_index = 0
            if feature == "franja_horaria" and "Noche" in options:
                default_index = options.index("Noche")
            row[feature] = c3.selectbox(feature, options, index=default_index)

        submitted = st.form_submit_button("Calcular probabilidad")

    if submitted:
        frame = pd.DataFrame([row])[metadata["numeric_features"] + metadata["categorical_features"]]
        probability = float(pipeline.predict_proba(frame)[0, 1])
        st.metric("Probabilidad estimada de accidente con victimas", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
