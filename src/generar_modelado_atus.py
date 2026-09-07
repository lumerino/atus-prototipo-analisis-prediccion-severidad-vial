"""Modelado supervisado de severidad binaria de accidentes de tránsito (ATUS, INEGI).

Genera la muestra de trabajo (streaming sobre los 28 CSV anuales), entrena y compara tres
clasificadores y escribe tablas/figuras en `resultados_modelado/`.

Variable objetivo
-----------------
`severidad_binaria` = 1 si el accidente tuvo víctimas (CLASACC = "Fatal" o "No fatal"),
0 si fue de "Sólo daños". Los registros cuya CLASACC no es clasificable (p. ej.
"Certificado cero") se descartan mediante `severity_from_clasacc(...) is None`.

Variables EXCLUIDAS explícitamente del conjunto de predictores
--------------------------------------------------------------
- `CLASACC`: define la etiqueta, usarla sería circular.
- Las 12 columnas de muertos/heridos (`CONDMUERTO`, `CONDHERIDO`, `PASAMUERTO`,
  `PASAHERIDO`, `PEATMUERTO`, `PEATHERIDO`, `CICLMUERTO`, `CICLHERIDO`, `OTROMUERTO`,
  `OTROHERIDO`, `NEMUERTO`, `NEHERIDO`): son la CONSECUENCIA del accidente y no se conocen
  de antemano; incluirlas sería fuga de información (data leakage) directa sobre la etiqueta.
- `COBERTURA`, `ESTATUS`, `ID_MINUTO`: metadatos administrativos, fuera del alcance.
- `ID_MUNICIPIO`: alta cardinalidad. NO es predictor del modelo base; sólo se conserva en los
  CSV de muestra para que `generar_sensibilidad_modelado.py` pueda construir `municipio_freq`
  en la prueba de sensibilidad correspondiente.

El `ColumnTransformer` usa `remainder="drop"` y listas explícitas de columnas, de modo que
ninguna columna fuera de `NUMERIC_FEATURES` + `CATEGORICAL_FEATURES` puede entrar al modelo.

La sección "Pruebas de sensibilidad" de `resumen_modelado.md` la agrega
`generar_sensibilidad_modelado.py`: si se regenera este resumen hay que volver a correr ese
script para recuperarla.

Uso: `py generar_modelado_atus.py`

NOTA (vendorizado para el prototipo del Entregable 4)
------------------------------------------------------
Esta es una copia literal de `actividad2/generar_modelado_atus.py`, incluida aquí
para que `entrenar_modelo.py` no dependa de la carpeta hermana `actividad2/` (ni al
importar, ni al des-serializar con joblib el `FunctionTransformer(to_dense, ...)`
del pipeline persistido, que referencia esta función por módulo). El prototipo
SOLO usa `build_features`, `build_preprocessor`, `load_sample`, `to_dense`,
`NUMERIC_FEATURES` y `CATEGORICAL_FEATURES`; el resto del archivo (streaming de
los 28 CSV crudos, entrenamiento de los 3 modelos, generación de figuras) no se
ejecuta aquí y sus constantes de ruta (`BASE_DIR`, `DATA_DIR`, `EDA_DIR`, `OUT_DIR`)
no son válidas en esta ubicación ni se usan.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Se reutilizan los auxiliares del EDA para que la limpieza y la definición de la etiqueta
# sean exactamente las mismas en ambas etapas (importar es seguro: el EDA sólo corre main()
# bajo `if __name__ == "__main__"`).
from generar_eda_atus import (
    HEAVY_VEHICLE_COLUMNS,
    VEHICLE_COLUMNS,
    VICTIM_COLUMNS,
    clean,
    norm_code,
    severity_from_clasacc,
    to_int,
    write_csv,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "actividad1" / "atus_zip" / "conjunto_de_datos"
EDA_DIR = BASE_DIR / "resultados_eda"
OUT_DIR = BASE_DIR / "resultados_modelado"

SEED = 42
TARGET_SAMPLE_SIZE = 750_000
TRAIN_VAL_YEARS = list(range(1997, 2024))
TEST_YEAR = 2024
PERMUTATION_SAMPLE_SIZE = 30_000
PERMUTATION_REPEATS = 5
# La importancia por permutación se mide con F1 de la clase positiva (NO con la exactitud por
# defecto): con ~78% de casos sin víctimas, la exactitud subestima las variables que sirven para
# detectar la clase minoritaria, que es justo el objetivo del modelo.
PERMUTATION_SCORING = "f1"
POBLACION_PROP_VICTIMAS = 0.2216  # resumen_eda_atus.md (10,666,069 accidentes válidos)

# Con False se reutilizan los CSV intermedios ya escritos (el muestreo es determinista con SEED,
# así que ambos caminos producen la misma muestra). Aun así se regeneran automáticamente si
# faltan o si su encabezado no coincide con SAMPLE_COLUMNS.
REGENERAR_MUESTRAS = False

# Columnas categóricas tomadas tal cual del CSV original.
RAW_CATEGORICAL_COLUMNS = [
    "MES",
    "DIASEMANA",
    "ID_ENTIDAD",
    "URBANA",
    "SUBURBANA",
    "TIPACCID",
    "CAUSAACCI",
    "CAPAROD",
    "SEXO",
    "ALIENTO",
    "CINTURON",
]

# FIX vendorizado (no presente en actividad2/generar_modelado_atus.py): el CSV crudo de ATUS
# trae DIASEMANA con variantes de mayúsculas/acentos inconsistentes entre años ("lunes" vs
# "Lunes", "Miercoles" vs "Miércoles", "Sabado" vs "Sábado"), que el `clean()` original no
# normaliza. Sin este mapeo, el OneHotEncoder trata cada variante como una categoría distinta,
# fragmentando ~170,000 registros (ver datos/atus_prototipo.db) entre dos columnas por día en
# vez de una, y el simulador del prototipo mostraba días duplicados en el desplegable. Solo se
# corrige aquí, en la copia local del Entregable 4: no se modifica actividad2/, para no alterar
# los resultados ya reportados en los Entregables 2 y 3.
DIASEMANA_CANONICO = {
    "lunes": "Lunes",
    "martes": "Martes",
    "miercoles": "Miércoles",
    "miércoles": "Miércoles",
    "jueves": "Jueves",
    "viernes": "Viernes",
    "sabado": "Sábado",
    "sábado": "Sábado",
    "domingo": "Domingo",
}

# Columnas que se guardan en los CSV intermedios de muestra (crudas + etiqueta).
# `ID_MUNICIPIO` se guarda pero NO es predictor: sólo lo consume la prueba de sensibilidad.
SAMPLE_COLUMNS = (
    ["ANIO", "ID_DIA", "ID_HORA", "ID_EDAD", "ID_MUNICIPIO"]
    + RAW_CATEGORICAL_COLUMNS
    + VEHICLE_COLUMNS
    + ["severidad_binaria"]
)

NUMERIC_FEATURES = (
    ["ANIO", "ID_DIA", "ID_HORA", "edad_valida", "edad_no_especificada"]
    + VEHICLE_COLUMNS
    + ["total_vehiculos", "involucra_motocicleta", "involucra_bicicleta", "involucra_pesado"]
)

CATEGORICAL_FEATURES = [
    "MES",
    "DIASEMANA",
    "franja_horaria",
    "ID_ENTIDAD",
    "URBANA",
    "SUBURBANA",
    "TIPACCID",
    "CAUSAACCI",
    "CAPAROD",
    "SEXO",
    "ALIENTO",
    "CINTURON",
]

# Salvaguarda: ninguna de estas columnas puede aparecer entre los predictores.
LEAKAGE_COLUMNS = VICTIM_COLUMNS + ["CLASACC"]

METRIC_FIELDS = ["modelo", "conjunto", "accuracy", "precision", "recall", "f1", "roc_auc"]


# ---------------------------------------------------------------------------
# Muestreo en streaming
# ---------------------------------------------------------------------------


def read_year_series() -> dict[int, dict[str, int]]:
    """Lee `resultados_eda/serie_severidad_por_anio.csv` (conteos ya calculados en el EDA)."""
    path = EDA_DIR / "serie_severidad_por_anio.csv"
    series: dict[int, dict[str, int]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            year = int(row["anio"])
            series[year] = {
                "accidentes_validos": int(row["accidentes_validos"]),
                "accidentes_con_victimas": int(row["accidentes_con_victimas"]),
            }
    return series


def extract_sample_row(row: dict[str, str], file_path: Path, severity: int) -> dict[str, object]:
    """Extrae del registro crudo sólo las columnas necesarias para construir las features."""
    record: dict[str, object] = {
        "ANIO": clean(row.get("ANIO")) or file_path.stem[-4:],
        "ID_DIA": clean(row.get("ID_DIA")),
        "ID_HORA": clean(row.get("ID_HORA")),
        "ID_EDAD": clean(row.get("ID_EDAD")),
        "MES": norm_code(row.get("MES"), 2),
        "ID_ENTIDAD": norm_code(row.get("ID_ENTIDAD"), 2),
        "ID_MUNICIPIO": norm_code(row.get("ID_MUNICIPIO"), 3),
    }
    for col in RAW_CATEGORICAL_COLUMNS:
        record.setdefault(col, clean(row.get(col)))
    for col in VEHICLE_COLUMNS:
        record[col] = to_int(row.get(col)) or 0
    record["severidad_binaria"] = severity
    return record


def stream_sample(files: list[Path], out_path: Path, fraction: float, rng: random.Random | None) -> tuple[int, int, int]:
    """Recorre los CSV anuales una sola vez y escribe la muestra en streaming.

    Devuelve (registros_validos_leidos, registros_escritos, registros_con_victimas_escritos).
    """
    valid_read = 0
    written = 0
    written_victims = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for file_path in files:
            file_written = 0
            with file_path.open("r", newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    severity = severity_from_clasacc(row.get("CLASACC"))
                    if severity is None:
                        continue
                    valid_read += 1
                    if rng is not None and not (rng.random() < fraction):
                        continue
                    writer.writerow(extract_sample_row(row, file_path, severity))
                    written += 1
                    file_written += 1
                    written_victims += severity
            print(f"  {file_path.name}: {file_written:,} filas muestreadas", flush=True)
    return valid_read, written, written_victims


def muestras_reutilizables(paths: list[Path]) -> bool:
    """True si los CSV de muestra existen y su encabezado coincide con SAMPLE_COLUMNS."""
    for path in paths:
        if not path.exists():
            return False
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), None)
        if header != SAMPLE_COLUMNS:
            return False
    return True


def generar_muestras() -> dict[str, object]:
    series = read_year_series()
    total_valid_train = sum(series[year]["accidentes_validos"] for year in TRAIN_VAL_YEARS)
    fraction = min(1.0, TARGET_SAMPLE_SIZE / total_valid_train)
    esperado_train = total_valid_train * fraction
    esperado_test = series[TEST_YEAR]["accidentes_validos"]

    train_files = [DATA_DIR / f"atus_anual_{year}.csv" for year in TRAIN_VAL_YEARS]
    test_files = [DATA_DIR / f"atus_anual_{TEST_YEAR}.csv"]
    for file_path in train_files + test_files:
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró {file_path}")

    train_path = OUT_DIR / "muestra_train_val_1997_2023.csv"
    test_path = OUT_DIR / "muestra_test_2024.csv"

    print(
        f"Accidentes válidos 1997-2023 (serie EDA): {total_valid_train:,} | "
        f"fracción de muestreo = {fraction:.6f} | esperado ~{esperado_train:,.0f} filas"
    )

    if REGENERAR_MUESTRAS or not muestras_reutilizables([train_path, test_path]):
        rng = random.Random(SEED)  # un solo generador para todo el pase 1997-2023
        print("Muestreo aleatorio 1997-2023 ...")
        valid_train, filas_train, victimas_train = stream_sample(train_files, train_path, fraction, rng)
        print("Pase completo 2024 (sin muestreo) ...")
        valid_test, filas_test, victimas_test = stream_sample(test_files, test_path, 1.0, None)
        print(
            f"Train/val: {filas_train:,} filas de {valid_train:,} válidas leídas "
            f"(esperado ~{esperado_train:,.0f}, desvío {filas_train - esperado_train:+,.0f})"
        )
        print(
            f"Prueba 2024: {filas_test:,} filas de {valid_test:,} válidas leídas "
            f"(esperado {esperado_test:,}, desvío {filas_test - esperado_test:+,})"
        )
        if valid_train != total_valid_train:
            print(
                "  AVISO: los válidos leídos en 1997-2023 no coinciden con la serie del EDA "
                f"({valid_train:,} vs {total_valid_train:,})."
            )
        if filas_test != esperado_test:
            print(
                "  AVISO: las filas de prueba 2024 no coinciden con la serie del EDA "
                f"({filas_test:,} vs {esperado_test:,})."
            )
        print(f"  Proporción con víctimas en la muestra train/val: {victimas_train / filas_train:.4%}")
        print(f"  Proporción con víctimas en prueba 2024: {victimas_test / filas_test:.4%}")
    else:
        print("Reutilizando los CSV de muestra ya existentes (mismo muestreo determinista).")

    return {
        "train_path": train_path,
        "test_path": test_path,
        "fraction": fraction,
        "total_valid_train": total_valid_train,
        "esperado_train": esperado_train,
        "esperado_test": esperado_test,
    }


# ---------------------------------------------------------------------------
# Ingeniería de variables
# ---------------------------------------------------------------------------


def franja_from_hora(hora: pd.Series) -> pd.Series:
    """Misma regla que `generar_eda_atus.py`: 00-05, 06-11, 12-17, 18-23; 99/NaN sin dato."""
    band = pd.Series("No especificada", index=hora.index, dtype=object)
    band[(hora >= 0) & (hora <= 5)] = "Madrugada"
    band[(hora >= 6) & (hora <= 11)] = "Mañana"
    band[(hora >= 12) & (hora <= 17)] = "Tarde"
    band[(hora >= 18) & (hora <= 23)] = "Noche"
    return band


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las variables predictoras a partir de la muestra cruda."""
    features = pd.DataFrame(index=df.index)

    features["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce")

    id_dia = pd.to_numeric(df["ID_DIA"], errors="coerce")
    features["ID_DIA"] = id_dia.mask(id_dia == 32)  # 32 = día no especificado

    id_hora = pd.to_numeric(df["ID_HORA"], errors="coerce")
    features["ID_HORA"] = id_hora.mask(id_hora == 99)  # 99 = hora no especificada

    edad = pd.to_numeric(df["ID_EDAD"], errors="coerce")
    features["edad_valida"] = edad.where((edad >= 1) & (edad <= 98))
    features["edad_no_especificada"] = edad.isin([0, 99]).astype(int)

    for col in VEHICLE_COLUMNS:
        features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    features["total_vehiculos"] = features[VEHICLE_COLUMNS].sum(axis=1)
    features["involucra_motocicleta"] = (features["MOTOCICLET"] > 0).astype(int)
    features["involucra_bicicleta"] = (features["BICICLETA"] > 0).astype(int)
    features["involucra_pesado"] = (features[HEAVY_VEHICLE_COLUMNS] > 0).any(axis=1).astype(int)

    features["franja_horaria"] = franja_from_hora(id_hora)

    for col in RAW_CATEGORICAL_COLUMNS:
        texto = df[col].fillna("").astype(str).str.strip()
        if col == "DIASEMANA":
            texto = texto.apply(lambda v: DIASEMANA_CANONICO.get(v.lower(), v))
        features[col] = texto.replace("", "No especificado")

    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def load_sample_with_raw(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Devuelve (muestra cruda, features, etiqueta). La cruda la usa `generar_sensibilidad_modelado`."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    y = raw["severidad_binaria"].astype(int)
    return raw, build_features(raw), y


def load_sample(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    _, features, y = load_sample_with_raw(path)
    return features, y


# ---------------------------------------------------------------------------
# Preprocesamiento, entrenamiento y evaluación
# ---------------------------------------------------------------------------


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Preprocesador base. Las listas son parámetros para que las pruebas de sensibilidad
    (`generar_sensibilidad_modelado.py`) puedan variar el conjunto de features sin duplicar código."""
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_features),
        ],
        remainder="drop",  # explícito: nada fuera de las dos listas entra al modelo
    )


def to_dense(matrix: object) -> np.ndarray:
    """Densifica a float32.

    `HistGradientBoostingClassifier` no acepta matrices dispersas en scikit-learn 1.2, así que
    las tres matrices se densifican una sola vez para que los tres modelos vean exactamente
    la misma representación.
    """
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def pretty_feature_name(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def evaluate_model(model, X: np.ndarray, y: pd.Series, nombre: str, conjunto: str) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
    metrics = {
        "modelo": nombre,
        "conjunto": conjunto,
        "accuracy": round(float(accuracy_score(y, y_pred)), 6),
        "precision": round(float(precision_score(y, y_pred, pos_label=1, zero_division=0)), 6),
        "recall": round(float(recall_score(y, y_pred, pos_label=1, zero_division=0)), 6),
        "f1": round(float(f1_score(y, y_pred, pos_label=1, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y, scores)), 6),
    }
    return metrics, y_pred, scores


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def figura_matriz_confusion(cm: np.ndarray, nombre_modelo: str, path: Path) -> None:
    total = cm.sum()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    etiquetas = ["Sólo daños (0)", "Con víctimas (1)"]
    ax.set_xticks([0, 1], labels=etiquetas)
    ax.set_yticks([0, 1], labels=etiquetas)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title(f"Matriz de confusión - {nombre_modelo}\nPrueba temporal 2024 (n = {total:,})")
    umbral = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n{cm[i, j] / total:.2%}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > umbral else "#1f2933",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Registros")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figura_importancia(rows: list[dict[str, object]], nombre_modelo: str, path: Path) -> None:
    labels = [str(row["variable"]) for row in rows][::-1]
    values = [float(row["importancia_media"]) for row in rows][::-1]
    errors = [float(row["importancia_desv_estandar"]) for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels, values, xerr=errors, color="#2f6f73")
    ax.set_xlabel("Importancia por permutación (caída media del F1 de la clase con víctimas)")
    ax.set_title(f"Top {len(rows)} variables - {nombre_modelo}\n(permutación sobre el conjunto de validación, scoring = F1)")
    ax.grid(axis="x", color="#e5e7eb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figura_roc(curvas: dict[str, np.ndarray], y_true: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, scores in curvas.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa5b1", label="Azar (AUC = 0.500)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC sobre la prueba temporal 2024")
    ax.legend(loc="lower right")
    ax.grid(color="#e5e7eb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Resumen narrativo
# ---------------------------------------------------------------------------


def make_summary_markdown(
    info_muestras: dict[str, object],
    n_train: int,
    n_val: int,
    n_test: int,
    prop_train: float,
    prop_val: float,
    prop_test: float,
    metric_rows: list[dict[str, object]],
    mejor_modelo: str,
    mejor_metrics: dict[str, object],
    cm: np.ndarray,
    importancias: list[dict[str, object]],
) -> None:
    n_muestra = n_train + n_val
    lines = [
        "# Resumen del modelado de severidad (ATUS)",
        "",
        "Este archivo fue generado automáticamente por `generar_modelado_atus.py` para apoyar el",
        "Entregable 2. Variable objetivo: `severidad_binaria` (1 = accidente con víctimas, es decir",
        "CLASACC Fatal o No fatal; 0 = Sólo daños). Los registros no clasificables (p. ej.",
        "\"Certificado cero\") se excluyen.",
        "",
        "## Muestras de trabajo",
        "",
        f"- Accidentes válidos 1997-2023 en la población: {int(info_muestras['total_valid_train']):,}",
        f"- Fracción de muestreo aleatorio simple (semilla {SEED}): {float(info_muestras['fraction']):.6f}",
        f"- Muestra 1997-2023: {n_muestra:,} registros (esperado ~{float(info_muestras['esperado_train']):,.0f})",
        f"- Entrenamiento (80%): {n_train:,} registros",
        f"- Validación (20%): {n_val:,} registros",
        f"- Prueba temporal 2024 (censo completo del año, sin muestreo): {n_test:,} registros",
        "",
        "## Proporción de la clase positiva (accidentes con víctimas)",
        "",
        "| Conjunto | Registros | Proporción con víctimas |",
        "|---|---:|---:|",
        f"| Entrenamiento (1997-2023) | {n_train:,} | {prop_train:.2%} |",
        f"| Validación (1997-2023) | {n_val:,} | {prop_val:.2%} |",
        f"| Prueba temporal 2024 | {n_test:,} | {prop_test:.2%} |",
        f"| Población 1997-2024 (EDA) | 10,666,069 | {POBLACION_PROP_VICTIMAS:.2%} |",
        "",
        "Las proporciones de las muestras 1997-2023 son consistentes con la proporción poblacional",
        f"({POBLACION_PROP_VICTIMAS:.2%}); la de 2024 es menor porque la proporción con víctimas",
        "viene bajando desde 1997 (28.87% en 1997 vs. 17.44% en 2024, ver",
        "`resultados_eda/serie_severidad_por_anio.csv`).",
        "",
        "## Variables excluidas para evitar fuga de información",
        "",
        f"- `CLASACC` (define la etiqueta) y las {len(VICTIM_COLUMNS)} columnas de muertos/heridos "
        f"({', '.join('`' + col + '`' for col in VICTIM_COLUMNS)}): son la consecuencia del",
        "  accidente y no se conocen antes de que ocurra.",
        "- `COBERTURA`, `ESTATUS`, `ID_MINUTO` (metadatos) y `ID_MUNICIPIO` (alta cardinalidad,",
        "  reservada para una prueba de sensibilidad posterior).",
        "",
        "El preprocesamiento (`ColumnTransformer` con `remainder='drop'`) se ajusta únicamente con el",
        "conjunto de entrenamiento y se aplica sin reajuste a validación y a la prueba 2024.",
        "El desbalance se corrige con `sample_weight` balanceado en los tres modelos.",
        "",
        "## Comparación de modelos",
        "",
        "| Modelo | Conjunto | Exactitud | Precisión | Sensibilidad | F1 | ROC-AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['modelo']} | {row['conjunto']} | {float(row['accuracy']):.4f} | "
            f"{float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{float(row['f1']):.4f} | {float(row['roc_auc']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Modelo seleccionado",
            "",
            f"**{mejor_modelo}**, por tener el F1-score más alto sobre la prueba temporal 2024",
            "(criterio fijo de selección; el ROC-AUC de la misma partición actúa como desempate).",
            "",
            f"- Exactitud: {float(mejor_metrics['accuracy']):.4f}",
            f"- Precisión (clase con víctimas): {float(mejor_metrics['precision']):.4f}",
            f"- Sensibilidad (clase con víctimas): {float(mejor_metrics['recall']):.4f}",
            f"- F1: {float(mejor_metrics['f1']):.4f}",
            f"- ROC-AUC: {float(mejor_metrics['roc_auc']):.4f}",
            "",
            "Matriz de confusión sobre la prueba 2024:",
            "",
            "| Real \\ Predicción | Sólo daños (0) | Con víctimas (1) |",
            "|---|---:|---:|",
            f"| Sólo daños (0) | {cm[0, 0]:,} | {cm[0, 1]:,} |",
            f"| Con víctimas (1) | {cm[1, 0]:,} | {cm[1, 1]:,} |",
            "",
            "## Variables más importantes",
            "",
            f"Importancia por permutación ({PERMUTATION_REPEATS} repeticiones) medida sobre el conjunto de",
            f"validación con `scoring='{PERMUTATION_SCORING}'` (F1 de la clase con víctimas), no con la exactitud",
            "por defecto: con ~78% de accidentes sin víctimas, la exactitud subestima las variables útiles",
            "para detectar la clase minoritaria. Se usa permutación en lugar de `feature_importances_` o",
            "`coef_` para que el criterio sea comparable entre modelos.",
            "",
            "| # | Variable | Importancia media | Desv. estándar |",
            "|---:|---|---:|---:|",
        ]
    )
    for i, row in enumerate(importancias[:5], start=1):
        lines.append(
            f"| {i} | {row['variable']} | {float(row['importancia_media']):.6f} | "
            f"{float(row['importancia_desv_estandar']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Archivos generados",
            "",
            "- `muestra_train_val_1997_2023.csv`, `muestra_test_2024.csv`: muestras de trabajo.",
            "- `tabla_comparacion_modelos.csv`: métricas de los tres modelos en validación y 2024.",
            "- `tabla_matriz_confusion.csv`, `figura_matriz_confusion.png`.",
            "- `tabla_importancia_variables.csv`, `figura_importancia_variables.png`.",
            "- `figura_roc_curvas.png`: curvas ROC de los tres modelos sobre la prueba 2024.",
        ]
    )

    (OUT_DIR / "resumen_modelado.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------


def main() -> None:
    inicio = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Salvaguarda contra fuga de información.
    usadas = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    filtradas = usadas.intersection(LEAKAGE_COLUMNS)
    if filtradas:
        raise ValueError(f"Columnas con fuga de información entre los predictores: {sorted(filtradas)}")

    info_muestras = generar_muestras()

    print("Cargando muestras y construyendo variables ...", flush=True)
    X_muestra, y_muestra = load_sample(info_muestras["train_path"])
    X_test, y_test = load_sample(info_muestras["test_path"])
    print(f"  Muestra 1997-2023: {X_muestra.shape[0]:,} filas | prueba 2024: {X_test.shape[0]:,} filas")

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_muestra,
        y_muestra,
        test_size=0.2,
        stratify=y_muestra,
        random_state=SEED,
    )
    print(
        f"  Entrenamiento: {len(y_train):,} ({y_train.mean():.4%} con víctimas) | "
        f"Validación: {len(y_val):,} ({y_val.mean():.4%} con víctimas) | "
        f"Prueba 2024: {len(y_test):,} ({y_test.mean():.4%} con víctimas)"
    )

    print("Ajustando el preprocesamiento SOLO con el conjunto de entrenamiento ...", flush=True)
    preprocessor = build_preprocessor()
    X_train = to_dense(preprocessor.fit_transform(X_train_df))
    X_val = to_dense(preprocessor.transform(X_val_df))
    X_test_matrix = to_dense(preprocessor.transform(X_test))
    feature_names = [pretty_feature_name(name) for name in preprocessor.get_feature_names_out()]
    print(f"  Matriz de diseño: {X_train.shape[1]} columnas tras el one-hot")

    sample_weight = compute_sample_weight("balanced", y_train)

    modelos = {
        "Regresión logística": LogisticRegression(max_iter=1000, random_state=SEED),
        "Bosque aleatorio": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "Gradient boosting (HistGB)": HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
    }

    metric_rows: list[dict[str, object]] = []
    scores_test: dict[str, np.ndarray] = {}
    predicciones_test: dict[str, np.ndarray] = {}
    for nombre, modelo in modelos.items():
        t0 = time.time()
        print(f"Entrenando {nombre} ...", flush=True)
        modelo.fit(X_train, y_train, sample_weight=sample_weight)
        print(f"  entrenado en {time.time() - t0:.1f} s", flush=True)
        met_val, _, _ = evaluate_model(modelo, X_val, y_val, nombre, "validacion")
        met_test, pred_test, sc_test = evaluate_model(modelo, X_test_matrix, y_test, nombre, "prueba_2024")
        metric_rows.extend([met_val, met_test])
        scores_test[nombre] = sc_test
        predicciones_test[nombre] = pred_test
        print(f"  validación: F1={met_val['f1']:.4f} AUC={met_val['roc_auc']:.4f}")
        print(f"  prueba 2024: F1={met_test['f1']:.4f} AUC={met_test['roc_auc']:.4f}")

    write_csv(OUT_DIR / "tabla_comparacion_modelos.csv", metric_rows, METRIC_FIELDS)

    sospechosos = [row for row in metric_rows if float(row["roc_auc"]) > 0.99]
    if sospechosos:
        print("AVISO: ROC-AUC > 0.99 detectado, posible fuga de información:")
        for row in sospechosos:
            print(f"  {row['modelo']} / {row['conjunto']}: AUC={row['roc_auc']}")

    # Selección: mayor F1 en prueba_2024; desempate por ROC-AUC en prueba_2024.
    filas_test = [row for row in metric_rows if row["conjunto"] == "prueba_2024"]
    mejor_fila = max(filas_test, key=lambda row: (float(row["f1"]), float(row["roc_auc"])))
    mejor_modelo = str(mejor_fila["modelo"])
    print(f"Mejor modelo por F1 en prueba 2024: {mejor_modelo} (F1={mejor_fila['f1']})")

    cm = confusion_matrix(y_test, predicciones_test[mejor_modelo], labels=[0, 1])
    cm_rows = [
        {"real": real, "prediccion": pred, "conteo": int(cm[real, pred])}
        for real in (0, 1)
        for pred in (0, 1)
    ]
    write_csv(OUT_DIR / "tabla_matriz_confusion.csv", cm_rows, ["real", "prediccion", "conteo"])
    figura_matriz_confusion(cm, mejor_modelo, OUT_DIR / "figura_matriz_confusion.png")

    # Importancia por permutación sobre VALIDACIÓN (nunca sobre la prueba 2024).
    if X_val.shape[0] > PERMUTATION_SAMPLE_SIZE:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(X_val.shape[0], size=PERMUTATION_SAMPLE_SIZE, replace=False)
        X_perm, y_perm = X_val[idx], y_val.iloc[idx]
        print(f"Importancia por permutación sobre submuestra de validación ({PERMUTATION_SAMPLE_SIZE:,} filas) ...", flush=True)
    else:
        X_perm, y_perm = X_val, y_val
        print(f"Importancia por permutación sobre validación completa ({X_val.shape[0]:,} filas) ...", flush=True)
    t0 = time.time()
    perm = permutation_importance(
        modelos[mejor_modelo],
        X_perm,
        y_perm,
        scoring=PERMUTATION_SCORING,  # F1 de la clase positiva, no la exactitud por defecto
        n_repeats=PERMUTATION_REPEATS,
        random_state=SEED,
    )
    print(f"  calculada en {time.time() - t0:.1f} s")

    orden = np.argsort(perm.importances_mean)[::-1][:20]
    importancia_rows = [
        {
            "variable": feature_names[i],
            "importancia_media": round(float(perm.importances_mean[i]), 6),
            "importancia_desv_estandar": round(float(perm.importances_std[i]), 6),
        }
        for i in orden
    ]
    write_csv(
        OUT_DIR / "tabla_importancia_variables.csv",
        importancia_rows,
        ["variable", "importancia_media", "importancia_desv_estandar"],
    )
    figura_importancia(importancia_rows, mejor_modelo, OUT_DIR / "figura_importancia_variables.png")

    figura_roc(scores_test, y_test, OUT_DIR / "figura_roc_curvas.png")

    make_summary_markdown(
        info_muestras,
        len(y_train),
        len(y_val),
        len(y_test),
        float(y_train.mean()),
        float(y_val.mean()),
        float(y_test.mean()),
        metric_rows,
        mejor_modelo,
        mejor_fila,
        cm,
        importancia_rows,
    )

    print(f"Resultados escritos en: {OUT_DIR}")
    print(f"Tiempo total: {(time.time() - inicio) / 60:.1f} minutos")


if __name__ == "__main__":
    main()
