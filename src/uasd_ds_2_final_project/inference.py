"""
Módulo de Inferencia para la Aplicación Web
=============================================
Capa de lógica separada de la interfaz de usuario. Se encarga de:

1. Cargar el modelo entrenado por `scripts/train.py` (`models/best_model.joblib`).
2. Derivar del propio pipeline qué variables espera como entrada.
3. Construir el registro de entrada a partir de los datos capturados por el usuario.
4. Estimar la probabilidad de suscripción y la decisión del modelo.
5. Leer los artefactos de evaluación (`reports/metrics.json`).

Regla de diseño importante: este módulo NO importa `streamlit`. Toda la lógica
es comprobable con `pytest` sin necesidad de levantar la aplicación web.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from uasd_ds_2_final_project.data import (
    CLIENT_CATEGORICAL_FEATURES,
    CLIENT_NUMERIC_FEATURES,
    ECONOMIC_FEATURES,
    load_raw_data,
)

# ==============================================================================
# 1. RUTAS DEL PROYECTO Y CONSTANTES
# ==============================================================================


def _find_project_root() -> Path:
    """
    Localiza la raíz del proyecto (donde vive `pyproject.toml`).

    Se admite la variable de entorno `UASD_DS_2_PROJECT_ROOT` como sobreescritura
    explícita, útil cuando el paquete se instala como distribución (no editable)
    y `__file__` apunta a `site-packages` en lugar del repositorio.
    """
    override = os.environ.get("UASD_DS_2_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate

    # Respaldo: <raiz>/src/uasd_ds_2_final_project/inference.py -> <raiz>
    return here.parents[1]


ROOT_DIR = _find_project_root()

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
DEFAULT_METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "bank-additional.csv"
DEFAULT_REPORTS_DIR = ROOT_DIR / "reports"
TRAIN_SCRIPT_PATH = ROOT_DIR / "scripts" / "train.py"

# Variables que el usuario puede alimentar con un número entero.
INTEGER_FEATURES = {"age", "campaign", "pdays", "previous"}


# ==============================================================================
# 2. EXCEPCIONES DE DOMINIO
# ==============================================================================


class ModelArtifactNotFound(FileNotFoundError):
    """El modelo entrenado todavía no existe en disco."""


class ModelArtifactError(RuntimeError):
    """El modelo existe pero no se puede cargar o no cumple el contrato esperado."""


# ==============================================================================
# 3. CARGA DE ARTEFACTOS
# ==============================================================================


def load_model(path: Path | str = DEFAULT_MODEL_PATH) -> Pipeline:
    """
    Carga el pipeline serializado por `scripts/train.py`.

    Lanza:
    ------
    ModelArtifactNotFound
        Si el archivo no existe. El mensaje indica el comando exacto para generarlo.
    ModelArtifactError
        Si el archivo está corrupto, fue generado con otra versión de las librerías
        o no corresponde a un `Pipeline` de scikit-learn.
    """
    model_path = Path(path)

    if not model_path.exists():
        raise ModelArtifactNotFound(
            f"No se encontro el modelo entrenado en: {model_path}\n\n"
            "Genera el artefacto con: uv run python scripts/train.py"
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - se reenvía como error de dominio
        raise ModelArtifactError(
            f"No se pudo cargar el modelo desde {model_path}. "
            "El archivo puede estar corrupto o generado con otra version de las librerias.\n"
            "Re-genera el artefacto con: uv run python scripts/train.py"
        ) from exc

    if not isinstance(model, Pipeline):
        raise ModelArtifactError(
            f"El artefacto {model_path} no es un Pipeline de scikit-learn, "
            f"sino {type(model).__name__}. Re-generalo con: uv run python scripts/train.py"
        )

    return model


def load_metrics(path: Path | str = DEFAULT_METRICS_PATH) -> dict[str, Any]:
    """
    Lee `reports/metrics.json`.

    Devuelve un diccionario vacío si el archivo no existe o está corrupto, para que
    la aplicación pueda continuar funcionando sin la pestaña de desempeño.
    """
    metrics_path = Path(path)

    if not metrics_path.exists():
        return {}

    try:
        content = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    return content if isinstance(content, dict) else {}


# ==============================================================================
# 4. CONTRATO DE VARIABLES DEL MODELO
# ==============================================================================


def expected_feature_columns(model: Pipeline) -> list[str]:
    """
    Deriva del modelo entrenado la lista exacta de columnas que espera como entrada.

    No se puede fijar a mano porque el mejor modelo puede provenir del experimento
    con variables económicas o sin ellas: el preprocesador serializado conserva
    su lista de columnas, que es la fuente de verdad.
    """
    preprocessor = model.named_steps.get("preprocessor")
    columns = getattr(preprocessor, "feature_names_in_", None)

    if columns is None:
        raise ModelArtifactError(
            "El preprocesador del modelo no expone 'feature_names_in_'. "
            "No es posible determinar las variables de entrada requeridas."
        )

    return list(columns)


def uses_economic_features(model: Pipeline) -> bool:
    """Indica si el modelo cargado hace uso de las variables macroeconómicas."""
    return bool(set(expected_feature_columns(model)) & set(ECONOMIC_FEATURES))


# ==============================================================================
# 5. METADATOS DEL DATASET PARA CONSTRUIR EL FORMULARIO
# ==============================================================================


def load_dataset_metadata(data_path: Path | str = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """
    Extrae del dataset crudo la información necesaria para construir el formulario.

    Derivar los valores desde los datos (y no hardcodearlos) evita que la interfaz
    quede desalineada con el dataset real.

    Returns
    -------
    dict con las claves:
        categorical_options : valores únicos por variable categórica.
        categorical_defaults: valor modal por variable categórica (default del formulario).
        numeric_ranges     : (mínimo, máximo, mediana) por variable numérica.
    """
    df = load_raw_data(data_path)

    categorical_options: dict[str, list[str]] = {}
    categorical_defaults: dict[str, str] = {}

    for column in CLIENT_CATEGORICAL_FEATURES:
        series = df[column].dropna().astype(str)
        values = sorted(series.unique().tolist())
        categorical_options[column] = values

        modes = series.mode()
        categorical_defaults[column] = modes.iloc[0] if not modes.empty else values[0]

    numeric_ranges: dict[str, tuple[float, float, float]] = {}

    for column in CLIENT_NUMERIC_FEATURES + ECONOMIC_FEATURES:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        numeric_ranges[column] = (
            float(series.min()),
            float(series.max()),
            float(series.median()),
        )

    return {
        "categorical_options": categorical_options,
        "categorical_defaults": categorical_defaults,
        "numeric_ranges": numeric_ranges,
    }


def load_categorical_options(data_path: Path | str = DEFAULT_DATA_PATH) -> dict[str, list[str]]:
    """Valores únicos observados en cada variable categórica del dataset."""
    return load_dataset_metadata(data_path)["categorical_options"]


def load_categorical_defaults(data_path: Path | str = DEFAULT_DATA_PATH) -> dict[str, str]:
    """Valor más frecuente de cada variable categórica del dataset."""
    return load_dataset_metadata(data_path)["categorical_defaults"]


def load_numeric_ranges(data_path: Path | str = DEFAULT_DATA_PATH) -> dict[str, tuple[float, float, float]]:
    """Rango (mínimo, máximo, mediana) de cada variable numérica del dataset."""
    return load_dataset_metadata(data_path)["numeric_ranges"]


# ==============================================================================
# 6. PREDICCIÓN
# ==============================================================================


def build_input_frame(payload: dict[str, Any], expected_columns: list[str]) -> pd.DataFrame:
    """
    Construye el registro de entrada del modelo.

    Se ignoran las claves extra del formulario y se exige la presencia de todas las
    columnas que el modelo espera, en el orden en que fueron ajustadas.
    """
    missing = [column for column in expected_columns if column not in payload]

    if missing:
        raise KeyError(f"Faltan variables requeridas por el modelo: {missing}")

    row = {column: payload[column] for column in expected_columns}
    return pd.DataFrame([row], columns=expected_columns)


def predict_subscription(
    model: Pipeline,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Estima la probabilidad de suscripción para un perfil de cliente.

    Returns
    -------
    dict con las claves:
        probability     : probabilidad de que el cliente suscriba (0.0 - 1.0).
        prediction      : decisión del modelo (1 = suscribe, 0 = no suscribe).
        label           : descripción legible de la decisión.
        expected_columns: variables que el modelo utilizó para la predicción.
    """
    columns = expected_feature_columns(model)
    frame = build_input_frame(payload, columns)

    prediction = int(model.predict(frame)[0])

    try:
        probability = float(model.predict_proba(frame)[0][1])
    except (AttributeError, IndexError, ValueError):
        # Clasificadores sin probabilidades (o ajustados con una sola clase).
        probability = 0.5

    label = "Si se suscribe" if prediction == 1 else "No se suscribe"

    return {
        "probability": probability,
        "prediction": prediction,
        "label": label,
        "expected_columns": columns,
    }


# ==============================================================================
# 7. RESUMEN DE MÉTRICAS
# ==============================================================================


def summarize_metrics(metrics: dict[str, Any]) -> pd.DataFrame:
    """
    Convierte la estructura anidada de `reports/metrics.json` en una tabla plana.

    Formato largo: una fila por (experimento, modelo) con todas sus métricas,
    ordenada de mayor a menor `f1_macro`, que es el criterio con el que
    `scripts/train.py` selecciona el mejor modelo.
    """
    rows: list[dict[str, Any]] = []

    for experiment, models in metrics.items():
        if not isinstance(models, dict):
            continue
        for model_name, values in models.items():
            row: dict[str, Any] = {"experimento": experiment, "modelo": model_name}
            if isinstance(values, dict):
                row.update(values)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    if "f1_macro" in summary.columns:
        summary = summary.sort_values("f1_macro", ascending=False).reset_index(drop=True)

    return summary
