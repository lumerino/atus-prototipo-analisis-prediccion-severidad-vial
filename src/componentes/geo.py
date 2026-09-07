"""Utilidades para el mapa coropletico de entidades.

Fuente geografica: `assets/mx_estados.geojson`, copia local de
angelnmara/geojson (mexicoHigh.json, MIT), 32 features a nivel estado. Se
descargo una sola vez y se versiona aqui para que el prototipo no dependa de
internet en tiempo de ejecucion (mismo criterio que datos_fuente/).

El mapa NO usa ese GeoJSON directamente: `mark_geoshape` con datos GEOMETRICOS
en linea (`alt.Data(values=...)`, sea GeoJSON o TopoJSON) renderiza en blanco
o lanza "Cannot read properties of undefined (reading 'length')" en
Object.Polygon/MultiPolygon con la version de Vega-Lite que trae empaquetada
Streamlit 1.63 (confirmado probando ambos formatos). La unica combinacion que
funciona es TopoJSON cargado por URL, asi que:

1. `static/mx_estados.topojson` (generado desde el geojson, ver
   `convertir_a_topojson()`) se sirve como archivo estatico de Streamlit
   (requiere `[server] enableStaticServing = true` en .streamlit/config.toml).
2. La app lo referencia por URL relativa (`URL_TOPOJSON_ESTADOS`), y solo la
   tabla de valores para el `transform_lookup` (no la geometria) viaja como
   dato inline normal — eso si funciona sin problema.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
GEOJSON_PATH = ASSETS_DIR / "mx_estados.geojson"
TOPOJSON_STATIC_PATH = Path(__file__).resolve().parents[1] / "static" / "mx_estados.topojson"

# Ruta relativa que Streamlit expone para archivos en src/static/ cuando
# enableStaticServing=true. Vega la resuelve contra el origen de la pagina,
# asi que funciona igual en localhost que en cualquier otro host/puerto.
URL_TOPOJSON_ESTADOS = "app/static/mx_estados.topojson"
OBJETO_TOPOJSON = "estados"

# El GeoJSON/TopoJSON identifica cada estado con un codigo ISO 3166-2
# ("MX-AGU"), no con el ID_ENTIDAD de dos digitos que usa INEGI/ATUS. Mapeo
# 1:1 verificado contra datos_fuente/catalogos/tc_entidad.csv (32 entidades
# en ambos lados).
ID_ENTIDAD_A_ISO = {
    "01": "MX-AGU", "02": "MX-BCN", "03": "MX-BCS", "04": "MX-CAM",
    "05": "MX-COA", "06": "MX-COL", "07": "MX-CHP", "08": "MX-CHH",
    "09": "MX-CMX", "10": "MX-DUR", "11": "MX-GUA", "12": "MX-GRO",
    "13": "MX-HID", "14": "MX-JAL", "15": "MX-MEX", "16": "MX-MIC",
    "17": "MX-MOR", "18": "MX-NAY", "19": "MX-NLE", "20": "MX-OAX",
    "21": "MX-PUE", "22": "MX-QUE", "23": "MX-ROO", "24": "MX-SLP",
    "25": "MX-SIN", "26": "MX-SON", "27": "MX-TAB", "28": "MX-TAM",
    "29": "MX-TLA", "30": "MX-VER", "31": "MX-YUC", "32": "MX-ZAC",
}


def convertir_a_topojson() -> None:
    """Regenera static/mx_estados.topojson desde assets/mx_estados.geojson.

    Solo hace falta correrlo si se reemplaza el GeoJSON fuente; el archivo
    resultante ya esta commiteado. Requiere `pip install topojson`."""
    import topojson as tp

    with GEOJSON_PATH.open(encoding="utf-8") as f:
        gj = json.load(f)
    topo = tp.Topology(gj, prequantize=False, object_name=OBJETO_TOPOJSON).to_dict()
    TOPOJSON_STATIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TOPOJSON_STATIC_PATH.open("w", encoding="utf-8") as f:
        json.dump(topo, f)


@st.cache_data(show_spinner=False)
def cargar_geojson_estados() -> dict:
    """Devuelve el GeoJSON fuente (no se usa para el mapa, solo referencia/QA)."""
    with GEOJSON_PATH.open(encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    convertir_a_topojson()
    print(f"TopoJSON regenerado en: {TOPOJSON_STATIC_PATH}")
