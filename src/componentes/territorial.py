from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from .estilo import AZUL, SECUENCIAL_AZUL, section_title
from .geo import ID_ENTIDAD_A_ISO, OBJETO_TOPOJSON, URL_TOPOJSON_ESTADOS
from .utils import query


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _bar(frame: pd.DataFrame, campo: str, titulo_eje: str) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_bar(color=AZUL, cornerRadiusEnd=4)
        .encode(
            x=alt.X("accidentes_totales:Q", title="Accidentes totales", axis=alt.Axis(format=",.0f")),
            y=alt.Y(f"{campo}:N", sort="-x", title=titulo_eje, axis=alt.Axis(labelLimit=220)),
            tooltip=[
                alt.Tooltip(f"{campo}:N", title=titulo_eje),
                alt.Tooltip("accidentes_totales:Q", title="Accidentes", format=",.0f"),
                alt.Tooltip("porcentaje_total:Q", title="% del total", format=".2f"),
            ],
        )
        .properties(height=340)
    )


def _mapa_entidades(datos_mapa: pd.DataFrame, metrica: str, titulo_metrica: str, formato: str) -> alt.Chart:
    fuente = alt.Data(url=URL_TOPOJSON_ESTADOS, format=alt.DataFormat(type="topojson", feature=OBJETO_TOPOJSON))
    return (
        alt.Chart(fuente)
        .mark_geoshape(stroke="#ffffff", strokeWidth=0.6)
        .encode(
            color=alt.Color(
                f"{metrica}:Q",
                title=titulo_metrica,
                scale=alt.Scale(range=SECUENCIAL_AZUL),
                legend=alt.Legend(format=formato),
            ),
            tooltip=[
                alt.Tooltip("properties.name:N", title="Entidad"),
                alt.Tooltip("accidentes_totales:Q", title="Accidentes totales", format=",.0f"),
                alt.Tooltip("pct_con_victimas:Q", title="% con víctimas", format=".1%"),
            ],
        )
        .transform_lookup(
            lookup="properties.id",
            from_=alt.LookupData(datos_mapa, "iso_id", ["accidentes_totales", "pct_con_victimas"]),
        )
        .project(type="mercator")
        .properties(height=460)
    )


def render(_: dict[str, object]) -> None:
    section_title("🗺️", "Distribución territorial")
    st.caption(
        "Mapa por entidad (todas las entidades, base completa 1997-2024) y top 10 entidades/municipios "
        "por volumen total de accidentes (serie histórica del EDA)."
    )

    por_entidad = query(
        """
        SELECT ID_ENTIDAD AS id_entidad,
               COUNT(*) AS accidentes_totales,
               AVG(CAST(severidad_binaria AS INTEGER)) AS pct_con_victimas
        FROM hechos_accidentes
        GROUP BY ID_ENTIDAD
        """
    )
    por_entidad["iso_id"] = por_entidad["id_entidad"].map(ID_ENTIDAD_A_ISO)
    por_entidad = por_entidad.dropna(subset=["iso_id"])

    metrica_label = st.radio(
        "Colorear mapa por:",
        ["Accidentes totales", "% con víctimas"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if metrica_label == "Accidentes totales":
        mapa = _mapa_entidades(por_entidad, "accidentes_totales", "Accidentes totales", ",.0f")
    else:
        mapa = _mapa_entidades(por_entidad, "pct_con_victimas", "% con víctimas", ".0%")
    st.altair_chart(mapa, use_container_width=True)

    entidades = query(
        """
        SELECT entidad, accidentes_totales, porcentaje_total
        FROM resumen_territorial
        WHERE tabla_origen = 'tabla_07_top_entidades_total'
        ORDER BY CAST(accidentes_totales AS INTEGER) DESC
        LIMIT 10
        """
    )
    municipios = query(
        """
        SELECT entidad || ' / ' || municipio AS municipio, accidentes_totales, porcentaje_total
        FROM resumen_territorial
        WHERE tabla_origen = 'tabla_09_top_municipios_total'
        ORDER BY CAST(accidentes_totales AS INTEGER) DESC
        LIMIT 10
        """
    )
    entidades = _numeric(entidades, ["accidentes_totales", "porcentaje_total"])
    municipios = _numeric(municipios, ["accidentes_totales", "porcentaje_total"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Top 10 entidades**")
        st.altair_chart(_bar(entidades, "entidad", "Entidad"), use_container_width=True)
    with right:
        st.markdown("**Top 10 municipios**")
        st.altair_chart(_bar(municipios, "municipio", "Municipio"), use_container_width=True)

    with st.expander("Ver tablas de datos"):
        c1, c2 = st.columns(2)
        c1.dataframe(entidades, width="stretch", hide_index=True)
        c2.dataframe(municipios, width="stretch", hide_index=True)
