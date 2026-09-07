# Prototipo ATUS — Guía rápida para el equipo

Este paquete trae **todo ya construido** (base de datos y modelo incluidos):
no necesitas regenerar nada para verlo funcionando, solo instalar dependencias
y correr la app.

## 1. Requisitos

- **Python 3.11** (con el que se desarrolló y probó). Otras versiones 3.x
  probablemente funcionen, pero no están garantizadas.
- Windows, macOS o Linux — no hay nada específico de Windows en el código
  (los ejemplos de comandos abajo son para PowerShell; en macOS/Linux usa
  `python3` en vez de `py -3`, y `/` en vez de `\` en las rutas).
- Conexión a internet **solo la primera vez** que instales dependencias
  (para descargar los paquetes de Python, incluyendo Prophet/CmdStan).

## 2. Instalación (una sola vez)

Abre una terminal **dentro de esta carpeta** (`prototipo_atus/`) y corre:

```powershell
py -3 -m pip install -r requirements.txt
```

Esto instala Streamlit, scikit-learn, Altair, Prophet, etc. Prophet instala
`cmdstanpy`, que en su primer uso descarga un binario compilado de CmdStan
(requiere internet). Si esa descarga falla, no es grave: la sección
"Pronóstico agregado" de la app cae automáticamente a un método más simple y
lo indica en pantalla.

## 3. Arrancar la app

```powershell
py -3 -m streamlit run src\app.py
```

Se abre sola en el navegador en `http://localhost:8501`. Si no se abre sola,
entra manualmente a esa dirección.

**No hace falta correr** `construir_base_datos.py` ni `entrenar_modelo.py` —
ya están construidos y vienen incluidos en este paquete:
- `datos/atus_prototipo.db` (la base de datos SQLite, ~317 MB)
- `modelos/modelo_severidad_histgb.joblib` (el modelo entrenado)
- `modelos/metadata_modelo.json` (métricas y metadatos del modelo)

## 4. Si algo falla

- **"No se encontró la base SQLite" / "No se encontró el modelo"**: revisa que
  las carpetas `datos/` y `modelos/` se hayan descomprimido junto con todo lo
  demás (a veces los compresores de ZIP omiten archivos muy grandes si el
  disco se queda sin espacio a mitad de la descompresión — revisa espacio
  libre en disco).
- **Prophet no instala / falla la descarga de CmdStan**: ignóralo, la app
  sigue funcionando con el fallback lineal en "Pronóstico agregado".
- **El mapa de "Distribución territorial" sale en blanco**: asegúrate de que
  la carpeta `.streamlit/` (con `config.toml`, que trae
  `enableStaticServing = true`) también se haya copiado — sin eso el mapa no
  puede servir el archivo `static/mx_estados.topojson` que necesita.
- **Puerto 8501 ocupado**: corre `py -3 -m streamlit run src\app.py --server.port 8502`
  (o cualquier otro puerto libre).

## 5. Si quieres regenerar todo desde cero (opcional, no necesario)

Este paquete también trae las muestras cacheadas originales en
`datos_fuente/resultados_modelado/` (~206 MB), así que si quieres reconstruir
la base y reentrenar el modelo tú mismo (por ejemplo, después de modificar el
código), puedes correr, en este orden:

```powershell
py -3 src\construir_base_datos.py
py -3 src\entrenar_modelo.py
```

Ambos scripts sobrescriben `datos/atus_prototipo.db` y
`modelos/modelo_severidad_histgb.joblib` respectivamente. Tarda unos minutos
(el entrenamiento del modelo es lo más lento).

## 6. Estructura de esta carpeta (referencia rápida)

```
prototipo_atus/
├── LEEME_EQUIPO.md          ← este archivo
├── README.md                 ← documentación técnica completa (arquitectura, hallazgos, etc.)
├── requirements.txt
├── .streamlit/config.toml    ← tema visual + enableStaticServing (necesario para el mapa)
├── src/
│   ├── app.py                ← punto de entrada de la app
│   ├── construir_base_datos.py
│   ├── entrenar_modelo.py
│   ├── componentes/          ← un módulo por sección del dashboard
│   ├── assets/                ← logo UNIR, GeoJSON fuente del mapa
│   └── static/                ← TopoJSON servido para el mapa
├── datos/atus_prototipo.db   ← base de datos ya construida
├── modelos/                  ← modelo ya entrenado + metadata
├── datos_fuente/              ← insumos originales (para regenerar, opcional)
└── capturas/                  ← capturas de referencia de la app
```

## 7. Más detalle

Para arquitectura completa, el modelo de datos, hallazgos de calidad de datos
encontrados en el camino, y cómo está pensado el simulador, lee `README.md` en
esta misma carpeta.

## 8. Repositorio en línea (opcional)

El código también vive en GitHub (público):
https://github.com/lumerino/atus-prototipo-analisis-prediccion-severidad-vial

Si prefieres clonarlo en vez de usar este ZIP, ten en cuenta que
`datos/atus_prototipo.db` está versionado con **Git LFS** — necesitas tener
`git-lfs` instalado (`git lfs install`) *antes* de clonar, o el archivo se
descargará como un puntero de texto en vez del archivo real.
