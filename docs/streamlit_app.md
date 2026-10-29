# Aplicación Streamlit — Predicción de suscripciones a depósitos a plazo

Interfaz web que consume el pipeline de entrenamiento del proyecto y permite estimar,
en vivo, la probabilidad de que un cliente potencial suscriba un depósito a plazo.

Resuelve el [Issue #10](https://github.com/yuniorlaureano/uasd-ds-2-final-project/issues/10)
("Crear aplicación de Streamlit").

---

## 1. Qué hace

| Pestaña | Contenido |
|---|---|
| **Predicción** | Formulario con el perfil del cliente (variables de contacto, perfil y contexto económico). Al enviarlo devuelve la probabilidad de suscripción y la decisión del modelo. |
| **Desempeño del modelo** | Tabla comparativa de los 6 experimentos (2 condiciones × 3 modelos), matriz de confusión y curvas ROC. |
| **Ficha del modelo** | Significado de cada variable, supuestos del modelado y limitaciones. |

La lógica de negocio vive en `src/uasd_ds_2_final_project/inference.py`, que **no importa
`streamlit`**. De ese modo toda la lógica es comprobable con `pytest` sin levantar la
aplicación, y la capa de presentación queda aislada en `src/uasd_ds_2_final_project/app.py`.

---

## 2. Requisitos

- Python `>= 3.12` (uv descarga la versión exacta definida en `.python-version`).
- [uv](https://docs.astral.sh/uv/) para instalar dependencias y ejecutar el proyecto.

---

## 3. Puesta en marcha

```bash
# 1. Instalar dependencias (crea el entorno .venv y genera uv.lock)
uv sync

# 2. Generar los artefactos del modelo (modelo + métricas + gráficos)
uv run python scripts/train.py

# 3. Levantar la aplicación
uv run streamlit run src/uasd_ds_2_final_project/app.py
```

La aplicación queda disponible en `http://localhost:8501`.

> **El paso 2 es obligatorio la primera vez.** La aplicación consume
> `models/best_model.joblib`, que `scripts/train.py` genera. Si el archivo no existe, la
> interfaz lo explica y ofrece un botón **"Entrenar el modelo ahora"** que ejecuta el
> entrenamiento sin salir de la aplicación (puede tardar entre 1 y 3 minutos).

Ejecución en modo headless (servidores, contenedores):

```bash
uv run streamlit run src/uasd_ds_2_final_project/app.py \
    --server.headless true --server.port 8501
```

---

## 4. Artefactos que consume

| Archivo | Generado por | Uso en la aplicación |
|---|---|---|
| `models/best_model.joblib` | `scripts/train.py` | Pipeline completo (preprocesador + clasificador) con el mejor `f1_macro`. |
| `reports/metrics.json` | `scripts/train.py` | Métricas de los 6 experimentos mostradas en la pestaña de desempeño. |
| `reports/confusion_matrix.png` | `scripts/train.py` | Matriz de confusión del mejor modelo. |
| `reports/roc_curves.png` | `scripts/train.py` | Curvas ROC comparativas. |
| `data/raw/bank-additional.csv` | Repositorio | Valores válidos y rangos de las variables del formulario. |

`models/*.joblib` está en `.gitignore`: es un binario grande y reproducible, no debe
versionarse.

---

## 5. Decisiones de diseño relevantes

1. **Las variables de entrada se derivan del modelo, no se fijan a mano.**
   `models/best_model.joblib` puede provenir del experimento *con* o *sin* variables
   económicas. La aplicación lee `preprocessor.feature_names_in_` del pipeline para saber
   exactamente qué columnas espera, de modo que funciona con cualquiera de los dos
   artefactos sin cambios de código.

2. **Las opciones del formulario se derivan del dataset.** Los valores de los selectores y
   los rangos (mínimo, máximo, mediana) de los campos numéricos se leen de
   `data/raw/bank-additional.csv`. Así la interfaz nunca ofrece un valor que el modelo no
   haya visto en entrenamiento, y los valores por defecto son representativos.

3. **La aplicación nunca se rompe por un artefacto ausente o corrupto.** Si falta el
   modelo se explica cómo generarlo; si está corrupto o fue guardado con otra versión de
   las librerías se pide re-entrenar. Si faltan las métricas, la pestaña de desempeño
   simplemente informa que no hay datos.

4. **La probabilidad mostrada se acompaña de su contexto.** Los modelos usan
   `class_weight='balanced'` para compensar el desbalance de clases (≈89% de clientes que no
   suscriben), por lo que el umbral 0.5 no equivale a la tasa real de conversión (≈11%).
   La interfaz lo advierte en lugar de presentar el número como si fuera la probabilidad
   de negocio.

---

## 6. Pruebas

```bash
uv run pytest
```

`tests/test_inference.py` cubre el contrato que consume la aplicación (carga del artefacto,
derivación de variables, construcción del registro de entrada, predicción, metadatos del
dataset y resumen de métricas) sin necesitar `streamlit` ni un modelo previamente
entrenado.

---

## 7. Estructura de archivos relacionadas

```
src/uasd_ds_2_final_project/
├── data.py        # carga, validación y preprocesamiento (propiedad del issue #7)
├── models.py      # fábricas de pipelines (propiedad del issue #7)
├── inference.py   # lógica de inferencia de la aplicación (este issue)
└── app.py         # interfaz Streamlit (este issue)
scripts/train.py   # entrenamiento y generación de artefactos (propiedad del issue #7)
```
