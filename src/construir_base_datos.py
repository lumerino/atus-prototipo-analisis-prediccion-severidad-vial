"""Construye el modelo de datos SQLite del prototipo ATUS.

El script reutiliza las muestras cacheadas del Entregable 2. No procesa los
CSV crudos anuales de ATUS.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEMINARIO_DIR = Path(__file__).resolve().parents[3]
ACTIVIDAD1 = SEMINARIO_DIR / "actividad1"
ACTIVIDAD2 = SEMINARIO_DIR / "actividad2"
ACTIVIDAD3 = SEMINARIO_DIR / "actividad3"

MODELADO_DIR = ACTIVIDAD2 / "resultados_modelado"
EDA_DIR = ACTIVIDAD2 / "resultados_eda"
CATALOGOS_DIR = ACTIVIDAD1 / "atus_zip" / "catalogos"
ADICIONALES_DIR = (
    ACTIVIDAD3
    / "TFM Entregable 3"
    / "TFM Entregable 3"
    / "Entregable 3 - Paquete final"
    / "resultados_modelado_adicionales"
)

DB_PATH = ROOT / "datos" / "atus_prototipo.db"
TRAIN_PATH = MODELADO_DIR / "muestra_train_val_1997_2023.csv"
TEST_PATH = MODELADO_DIR / "muestra_test_2024.csv"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el insumo esperado: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def write_sample_table(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS hechos_accidentes")
    first = True
    for path, conjunto in ((TRAIN_PATH, "train_val"), (TEST_PATH, "test_2024")):
        for chunk in pd.read_csv(path, encoding="utf-8-sig", dtype=str, chunksize=100_000):
            chunk["conjunto"] = conjunto
            chunk.to_sql(
                "hechos_accidentes",
                conn,
                if_exists="replace" if first else "append",
                index=False,
            )
            first = False
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hechos_anio ON hechos_accidentes(ANIO)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hechos_entidad ON hechos_accidentes(ID_ENTIDAD)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hechos_municipio ON hechos_accidentes(ID_ENTIDAD, ID_MUNICIPIO)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hechos_tipaccid ON hechos_accidentes(TIPACCID)")


def write_catalogs(conn: sqlite3.Connection) -> None:
    read_csv(CATALOGOS_DIR / "tc_entidad.csv", dtype=str).to_sql(
        "dim_entidad", conn, if_exists="replace", index=False
    )
    read_csv(CATALOGOS_DIR / "tc_municipio.csv", dtype=str).to_sql(
        "dim_municipio", conn, if_exists="replace", index=False
    )


def write_eda_summaries(conn: sqlite3.Connection) -> None:
    read_csv(EDA_DIR / "serie_severidad_por_anio.csv").to_sql(
        "resumen_anual", conn, if_exists="replace", index=False
    )
    territorial_files = [
        "tabla_07_top_entidades_total.csv",
        "tabla_08_top_entidades_severidad.csv",
        "tabla_09_top_municipios_total.csv",
        "tabla_10_top_municipios_severidad.csv",
    ]
    frames = []
    for filename in territorial_files:
        frame = read_csv(EDA_DIR / filename, dtype=str)
        frame["tabla_origen"] = filename.removesuffix(".csv")
        frames.append(frame)
    pd.concat(frames, ignore_index=True, sort=False).to_sql(
        "resumen_territorial", conn, if_exists="replace", index=False
    )


def write_model_comparison(conn: sqlite3.Connection) -> None:
    paths = [
        MODELADO_DIR / "tabla_comparacion_modelos.csv",
        MODELADO_DIR / "tabla_comparacion_modelos_ampliada.csv",
        ADICIONALES_DIR / "tabla_comparacion_modelos_ampliada.csv",
    ]
    frames = [read_csv(path) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No se encontro ninguna tabla de comparacion de modelos.")
    comparison = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["modelo", "conjunto"], keep="last"
    )
    comparison.to_sql("comparacion_modelos", conn, if_exists="replace", index=False)


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        write_sample_table(conn)
        write_catalogs(conn)
        write_eda_summaries(conn)
        write_model_comparison(conn)
        conn.execute(
            "CREATE VIEW IF NOT EXISTS v_hechos_con_entidad AS "
            "SELECT h.*, e.NOM_ENTIDAD "
            "FROM hechos_accidentes h "
            "LEFT JOIN dim_entidad e ON printf('%02d', CAST(h.ID_ENTIDAD AS INTEGER)) = e.ID_ENTIDAD"
        )
    print(f"Base SQLite creada en: {DB_PATH}")


if __name__ == "__main__":
    main()
