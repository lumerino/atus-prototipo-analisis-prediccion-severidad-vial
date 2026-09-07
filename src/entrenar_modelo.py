"""Entrena y persiste el modelo ganador del proyecto ATUS.

Usa la copia local vendorizada de `generar_modelado_atus.py` (en esta misma
carpeta `src/`), no la de `actividad2/`, para que el prototipo sea
autocontenido y el pipeline persistido (que referencia `to_dense` por módulo
al des-serializarse con joblib) no dependa de una carpeta hermana.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight

from generar_modelado_atus import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features, build_preprocessor, load_sample, to_dense


ROOT = Path(__file__).resolve().parents[1]
MODELADO_DIR = ROOT / "datos_fuente" / "resultados_modelado"


TRAIN_PATH = MODELADO_DIR / "muestra_train_val_1997_2023.csv"
TEST_PATH = MODELADO_DIR / "muestra_test_2024.csv"
MODEL_PATH = ROOT / "modelos" / "modelo_severidad_histgb.joblib"
METADATA_PATH = ROOT / "modelos" / "metadata_modelo.json"


def metrics_dict(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6),
    }


def categorical_values(*frames: pd.DataFrame) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for col in CATEGORICAL_FEATURES:
        series = pd.concat([frame[col] for frame in frames], ignore_index=True).dropna().astype(str)
        values[col] = sorted(series.unique().tolist())
    return values


def main() -> None:
    ROOT.joinpath("modelos").mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_sample(TRAIN_PATH)
    X_test, y_test = load_sample(TEST_PATH)

    pipeline = Pipeline(
        steps=[
            ("preprocesador", build_preprocessor()),
            ("densificador", FunctionTransformer(to_dense, accept_sparse=True)),
            ("clasificador", HistGradientBoostingClassifier(max_iter=200, random_state=42)),
        ]
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(X_train, y_train, clasificador__sample_weight=weights)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metadata = {
        "modelo": "HistGradientBoostingClassifier",
        "descripcion": "Pipeline con preprocesador oficial del Entregable 2 y clasificador HistGB.",
        "entrenamiento": "Muestra cacheada 1997-2023 completa",
        "evaluacion": "Censo 2024 cacheado",
        "metricas_prueba_2024": metrics_dict(y_test, y_pred, y_score),
        "matriz_confusion": cm,
        "roc_curve": {
            "fpr": [round(float(v), 6) for v in fpr[:: max(1, len(fpr) // 250)]],
            "tpr": [round(float(v), 6) for v in tpr[:: max(1, len(tpr) // 250)]],
            "thresholds": [round(float(v), 6) for v in thresholds[:: max(1, len(thresholds) // 250)]],
        },
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categorical_values": categorical_values(X_train, X_test),
        "numeric_defaults": {col: float(pd.to_numeric(X_train[col], errors="coerce").median()) for col in NUMERIC_FEATURES},
    }

    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata["metricas_prueba_2024"], ensure_ascii=False, indent=2))
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Metadata guardada en: {METADATA_PATH}")


if __name__ == "__main__":
    main()
