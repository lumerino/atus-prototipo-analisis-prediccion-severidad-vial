"""Paleta y estilos compartidos por todas las vistas del dashboard.

Paleta categorica, secuencial y de estado tomadas de la guia interna de
visualizacion de datos del equipo (validada para daltonismo y contraste sobre
superficie clara). Un unico punto de definicion para que los seis componentes
usen exactamente los mismos colores.
"""

from __future__ import annotations

import altair as alt
import streamlit as st

# Paleta categorica (orden fijo, nunca ciclada libremente).
AZUL = "#2a78d6"
NARANJA = "#eb6834"
AGUA = "#1baf7a"
AMARILLO = "#eda100"
MAGENTA = "#e87ba4"
VERDE = "#008300"
VIOLETA = "#4a3aa7"
ROJO = "#e34948"

CATEGORICA = [AZUL, NARANJA, AGUA, AMARILLO, MAGENTA, VERDE, VIOLETA, ROJO]

# Rampa secuencial de un solo tono (azul), pasos 100 a 700, para magnitud.
SECUENCIAL_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Paleta de estado (fija, nunca para series categoricas).
ESTADO_BUENO = "#0ca30c"
ESTADO_ALERTA = "#fab219"
ESTADO_SERIO = "#ec835a"
ESTADO_CRITICO = "#d03b3b"

# Tinta y superficie.
TEXTO_PRIMARIO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
TEXTO_MUTED = "#898781"
GRILLA = "#e1e0d9"
SUPERFICIE = "#fcfcfb"


def probabilidad_color(valor: float) -> tuple[str, str]:
    """Devuelve (color, etiqueta) segun umbrales de riesgo para una probabilidad 0-1."""
    if valor < 0.30:
        return ESTADO_BUENO, "Riesgo bajo"
    if valor < 0.55:
        return ESTADO_ALERTA, "Riesgo medio"
    if valor < 0.75:
        return ESTADO_SERIO, "Riesgo alto"
    return ESTADO_CRITICO, "Riesgo muy alto"


def registrar_tema_altair() -> None:
    def tema() -> dict:
        return {
            "config": {
                "background": SUPERFICIE,
                "title": {"color": TEXTO_PRIMARIO, "fontSize": 15, "font": "system-ui, -apple-system, Segoe UI, sans-serif", "anchor": "start"},
                "axis": {
                    "labelColor": TEXTO_MUTED,
                    "titleColor": TEXTO_SECUNDARIO,
                    "gridColor": GRILLA,
                    "domainColor": "#c3c2b7",
                    "tickColor": "#c3c2b7",
                    "labelFont": "system-ui, -apple-system, Segoe UI, sans-serif",
                    "titleFont": "system-ui, -apple-system, Segoe UI, sans-serif",
                    "gridDash": [1, 0],
                },
                "legend": {
                    "labelColor": TEXTO_SECUNDARIO,
                    "titleColor": TEXTO_SECUNDARIO,
                    "labelFont": "system-ui, -apple-system, Segoe UI, sans-serif",
                },
                "view": {"stroke": "transparent"},
                "range": {"category": CATEGORICA},
            }
        }

    alt.themes.register("atus", tema)
    alt.themes.enable("atus")


def inyectar_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}

        html, body, [class*="css"] { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }

        .stAppHeader { background: transparent; }

        [data-testid="stSidebar"] {
            background-color: #f2f3f5;
            border-right: 1px solid #e1e0d9;
        }
        [data-testid="stSidebar"] h1 { font-size: 1.05rem; }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e1e0d9;
            border-left: 4px solid #2a78d6;
            border-radius: 10px;
            padding: 0.9rem 1rem 0.7rem 1rem;
            box-shadow: 0 1px 2px rgba(11,11,11,0.04);
        }
        div[data-testid="stMetric"] label { color: #52514e !important; font-weight: 600; }
        div[data-testid="stMetricValue"] { color: #0b0b0b; }

        .atus-hero {
            background: linear-gradient(120deg, #14335c 0%, #2a78d6 100%);
            color: #ffffff;
            padding: 1.6rem 1.8rem;
            border-radius: 14px;
            margin-bottom: 1.4rem;
        }
        .atus-hero h1 { color: #ffffff; margin: 0 0 0.25rem 0; font-size: 1.55rem; }
        .atus-hero p { color: #d7e6fb; margin: 0; font-size: 0.92rem; }

        .atus-section-title {
            display: flex; align-items: center; gap: 0.5rem;
            font-size: 1.2rem; font-weight: 700; color: #0b0b0b;
            border-left: 5px solid #2a78d6;
            padding-left: 0.6rem;
            margin: 0.4rem 0 0.9rem 0;
        }

        .atus-card {
            background: #ffffff;
            border: 1px solid #e1e0d9;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 1px 2px rgba(11,11,11,0.04);
        }

        .atus-badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            color: #ffffff;
        }

        .atus-caption { color: #898781; font-size: 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_title(icon: str, text: str) -> None:
    st.markdown(f'<div class="atus-section-title">{icon} {text}</div>', unsafe_allow_html=True)
