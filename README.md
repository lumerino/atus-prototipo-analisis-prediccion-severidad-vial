# Prototipo ATUS - Entregable 4

Aplicacion local en Streamlit para recrear los principales resultados del TFM
sobre severidad de accidentes de transito en Mexico con la base ATUS de INEGI
(1997-2024). El prototipo integra un modelo de datos SQLite, visualizaciones de
EDA y un simulador predictivo basado en `HistGradientBoostingClassifier`.

## Instalacion

```powershell
py -3 -m pip install -r requirements.txt
```

## Ejecucion reproducible

Desde esta carpeta:

```powershell
py -3 src\construir_base_datos.py
py -3 src\entrenar_modelo.py
py -3 -m streamlit run src\app.py
```

La app queda disponible normalmente en `http://localhost:8501`.

## Arquitectura

- `src/construir_base_datos.py`: crea `datos/atus_prototipo.db` reutilizando las
  muestras cacheadas de `actividad2/resultados_modelado/`. No reprocesa los CSV
  crudos de `actividad1/atus_zip/`.
- `src/entrenar_modelo.py`: importa `build_features`, `build_preprocessor` y las
  listas oficiales de variables desde `actividad2/generar_modelado_atus.py`.
- `src/app.py`: punto de entrada Streamlit con navegacion lateral.
- `src/componentes/`: modulos de dashboard por seccion.
- `modelos/modelo_severidad_histgb.joblib`: pipeline entrenado localmente.
- `modelos/metadata_modelo.json`: metricas, matriz de confusion, curva ROC y
  valores auxiliares del simulador.
- `capturas/`: evidencia visual de las seis secciones de la app.

## Modelo de datos

La app lee desde `datos/atus_prototipo.db`; los CSV cacheados solo se usan para
generar la base. Tablas principales:

| Tabla | Contenido | Uso en el prototipo |
|---|---|---|
| `hechos_accidentes` | Muestras 1997-2023 y censo 2024 con `conjunto` y `severidad_binaria` | KPIs, filtros, perfiles y agregaciones |
| `dim_entidad` | Catalogo de entidades | Etiquetas territoriales |
| `dim_municipio` | Catalogo de municipios | Etiquetas municipales |
| `resumen_anual` | Serie anual de accidentes validos y con victimas | Tendencias y pronostico |
| `resumen_territorial` | Top entidades y municipios del EDA | Barras territoriales |
| `comparacion_modelos` | Comparacion de modelos base y adicionales | Contexto de desempeno |
| `v_hechos_con_entidad` | Vista de hechos con nombre de entidad | Consultas exploratorias |

## Modelo predictivo

El modelo final se entreno sobre la muestra cacheada 1997-2023 completa y se
evaluo contra el censo 2024. Metricas de prueba:

| Metrica | Valor |
|---|---:|
| Accuracy | 0.830199 |
| Precision | 0.509059 |
| Recall | 0.733908 |
| F1 | 0.601146 |
| ROC-AUC | 0.878848 |

Estas cifras son consistentes con el modelo ganador reportado en los entregables
2 y 3. La pequena variacion se debe a que aqui se ajusta el pipeline final sobre
todo 1997-2023, sin reservar de nuevo el subconjunto de validacion.

## Secciones de la app

1. `Resumen general`: KPIs globales y serie anual filtrada.
2. `Tendencias`: series historicas anuales y proporcion con victimas.
3. `Distribucion territorial`: top entidades y municipios.
4. `Perfil de severidad`: tipo de accidente, hora y variables del conductor.
5. `Modelo predictivo`: metricas, matriz de confusion, ROC y simulador.
6. `Pronostico agregado`: fallback lineal local sobre la serie anual.

## Capturas

![Resumen general](capturas/01_resumen_general.png)
![Tendencias](capturas/02_tendencias.png)
![Distribucion territorial](capturas/03_distribucion_territorial.png)
![Perfil de severidad](capturas/04_perfil_severidad.png)
![Modelo predictivo](capturas/05_modelo_predictivo.png)
![Pronostico agregado](capturas/06_pronostico_agregado.png)

## Notas de versionado

`datos/*.db` y `modelos/*.joblib` se excluyen de Git por ser artefactos
regenerables. El repositorio versiona el codigo, la metadata del modelo, el
README y las capturas usadas como evidencia del prototipo.
