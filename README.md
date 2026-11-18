# PSet 3 · Experimentación y selección del modelo

Pronóstico de ventas diarias de perecederos por tienda–producto, horizonte de
16 días (Corporación Favorita). Parte de la OBT publicada por el PSet 2 en
Snowflake (`PSET2_DB.ANALYTICS.OBT_SALES`) y termina con un modelo registrado
en MLflow como `favorita-perecederos-modelo@champion`.

```
OBT → EDA → Features → Pipeline → Línea base → Candidatos → Tuning → Selección → Model Registry
```

## Requisitos

- Docker Desktop (contenedores Linux).
- Python 3.12. Con [uv](https://docs.astral.sh/uv/):
  ```powershell
  uv venv --python 3.12 .venv
  uv pip install -r requirements.txt --python .venv\Scripts\python.exe --index-strategy unsafe-best-match
  ```
  Con pip: `python -m pip install -r requirements.txt` (el archivo incluye el
  índice de PyTorch para CPU, necesario para Chronos).
- La cuenta de Snowflake y la clave RSA configuradas en el PSet 2, con la OBT ya publicada.

## 1. Configuración

```powershell
Copy-Item .env.example .env
```

| Variable | Qué poner |
|---|---|
| `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER` | Los mismos valores del `.env` del PSet 2 |
| `SNOWFLAKE_ROLE`, `_WAREHOUSE`, `_DATABASE`, `_OBT_TABLE` | Dejar los del ejemplo (creados en el PSet 2) |
| `SNOWFLAKE_PRIVATE_KEY_FILE`, `_PASSPHRASE_FILE` | Rutas a la clave y la frase del PSet 2. Relativas a la raíz de este repositorio o absolutas; no copiar los secretos aquí |
| `MLFLOW_POSTGRES_PASSWORD` | Contraseña local nueva para la base de metadatos de MLflow |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5050` |

`.env` está en `.gitignore`: el repositorio no contiene contraseñas, claves ni tokens.

## 2. Levantar MLflow

```powershell
docker compose up -d --build
docker compose ps          # mlflow y mlflow-db deben estar "healthy"
```

Interfaz: http://127.0.0.1:5050

| Pieza | Implementación |
|---|---|
| Servidor de tracking | `mlflow server` 3.18.0 (misma versión que el cliente), imagen en `mlflow/Dockerfile` |
| Backend store | PostgreSQL 16.6 (`mlflow-db`), volumen `mlflow_db` |
| Artifact store | Volumen `mlflow_artifacts`, servido por el propio servidor (`--serve-artifacts`); el cliente sube los modelos por HTTP (`mlflow-artifacts:/`) |

El job runner de MLflow 3 (tareas GenAI) se desactiva con
`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`: lanzaba ~8 procesos y el
contenedor moría por memoria. Detener sin borrar el registro: `docker compose stop`.

## 3. Datos

```powershell
.\.venv\Scripts\python.exe -m src.datos
```

Extrae de la OBT una muestra de **2.000 series tienda–producto** (estratificada
y proporcional por familia, semilla 42, series con historia desde antes del
15/06/2017), completas en el tiempo, y la guarda en `data/obt_muestra.parquet`
con sus metadatos (`data/obt_muestra_meta.json`): tabla de origen, fecha de
corte del snapshot y huella SHA-256 del contenido. Si el archivo ya existe,
verifica su huella (calculada sobre el contenido leído del disco, con tipos normalizados). Los notebooks lo llaman automáticamente.

| Snapshot verificado | Valor |
|---|---|
| OBT de origen | 13.017.002 filas, 2016-08-01 a 2017-08-15 |
| Series elegibles / muestreadas | 37.545 / 2.000 |
| Filas de la muestra | 692.950 |
| SHA-256 | `4c43165b8f2b72904e5163ee431b3d9dee18bec1c820ee2a4deb035538e899cc` |

Dos extracciones independientes producen la misma huella.

## 4. Orden de ejecución de los notebooks

Ejecutar cada uno con **Restart** y **Run All**, en este orden, con MLflow levantado:

| Notebook | Etapa | Tiempo medido |
|---|---|---|
| `notebooks/01_eda.ipynb` | Exploración de la OBT y hallazgos | |
| `notebooks/02_features.ipynb` | Features derivadas, prueba de no fuga y ablación de promociones | |
| `notebooks/03_baseline_candidatos.ipynb` | Líneas base, 5 familias de candidatos y LLM por prompting | |
| `notebooks/04_tuning.ipynb` | Ajuste con Optuna de los 3 mejores candidatos (90 corridas hijas) | ~10 h |
| `notebooks/05_seleccion.ipynb` | Políticas de pedido, selección por métrica de negocio, prueba y Model Registry | ~1,5 h |
| `notebooks/06_reproducir_champion.ipynb` | Kernel limpio: carga `@champion` y reproduce sus métricas de prueba | |

El LLM del notebook 03 no usa API: el prompt (`docs/llm/prompt_llm.txt`) se pegó a mano en un chat y la respuesta
quedó en `docs/llm/respuesta_llm.json`. El notebook solo la evalúa.

## 5. Resultado

**Criterio de selección:** gana el modelo con la mejor **métrica de negocio**, el costo de la regla de reposición
*newsvendor* del PSet 1 (Co · sobrantes + Cu · faltantes, vida útil de un día), medido en la validación. El escenario
principal es Co:Cu = 1:3: faltar cuesta el triple que sobrar. Como el pedido óptimo es el cuantil
q = Cu/(Cu+Co) = 0,75 de la demanda, los modelos finales pronostican ese cuantil directamente.

| Alias en `favorita-perecederos-modelo` | Modelo | Costo prueba 1:3 vs media 28 días | Merma | Cobertura | RMSLE prueba |
|---|---|---|---|---|---|
| `@champion` | Chronos-Bolt (small) cuantil 0,75 | **−12,0 %** | 34,2 % | 89,1 % | 0,783 |
| `@challenger` | Gradient boosting con pérdida de cuantil 0,75 | −10,4 % | 34,4 % | 88,7 % | 0,808 |
| referencia | Gradient boosting ajustado por RMSLE | −0,4 % | 35,9 % | 86,3 % | 0,648 |
| referencia | Media de 28 días + stock de seguridad | — | 31,2 % | 82,5 % | 0,684 |

La salida del modelo registrado es el **pedido sugerido** en unidades por tienda–producto–día. Su RMSLE es mayor que
el del modelo por RMSLE a propósito: pide por encima de la mediana porque faltar cuesta más. El detalle y las
limitaciones están en el notebook 05 (§9) y en el memo.

## Estructura

| Ruta | Contenido |
|---|---|
| `src/datos.py` | Conexión a Snowflake, muestra de series y huella del snapshot |
| `src/pipeline.py` | Features, preparación (`Pipeline` + `ColumnTransformer`), división, validación temporal, métricas y estimadores propios (`ReglaSemanal`, `ChronosBolt`) |
| `notebooks/` | Etapas del experimento |
| `docs/corridas.csv` | Corridas del experimento exportadas con `mlflow.search_runs` |
| `docs/figuras/` | Figuras que generan los notebooks (también se registran en MLflow) |
| `docs/llm/` | Prompt, series y respuesta del LLM por prompting (notebook 03) |
| `docs/PSet3_memo_Rojas_Barazarte.pdf` | Documento técnico |
| `mlflow/` | Imagen del servidor de tracking |
