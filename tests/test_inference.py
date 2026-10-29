"""
Pruebas Unitarias para el Módulo de Inferencia
================================================
Verifica el contrato que consume la aplicación Streamlit:

1. Carga y validación del artefacto del modelo.
2. Derivación de las variables de entrada a partir del pipeline entrenado.
3. Construcción del registro de entrada y predicción.
4. Metadatos del dataset usados para construir el formulario.
5. Resumen de las métricas de evaluación.

Estas pruebas no dependen de `streamlit` ni de un modelo previamente entrenado:
construyen un pipeline mínimo con las fábricas que ya existen en `models.py`.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from uasd_ds_2_final_project.data import (
    CLIENT_CATEGORICAL_FEATURES,
    CLIENT_NUMERIC_FEATURES,
    ECONOMIC_FEATURES,
    build_preprocessor,
    prepare_features_and_target,
)
from uasd_ds_2_final_project.inference import (
    DEFAULT_DATA_PATH,
    ModelArtifactError,
    ModelArtifactNotFound,
    build_input_frame,
    expected_feature_columns,
    load_categorical_defaults,
    load_categorical_options,
    load_dataset_metadata,
    load_metrics,
    load_model,
    load_numeric_ranges,
    predict_subscription,
    summarize_metrics,
    uses_economic_features,
)
from uasd_ds_2_final_project.models import build_logistic_regression_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = REPO_ROOT / "data" / "raw" / "bank-additional.csv"


def _fit_sample_pipeline(include_economic_features: bool = True) -> Pipeline:
    """Ajusta un pipeline mínimo y utilizable como modelo entrenado de prueba."""
    data = {
        "age": [25, 45, 33, 58, 29, 41, 62, 35],
        "job": ["admin.", "technician", "services", "blue-collar", "admin.", "management", "retired", "technician"],
        "marital": ["single", "married", "married", "divorced", "single", "married", "married", "single"],
        "education": ["university.degree", "high.school", "basic.9y", "professional.course", "university.degree", "university.degree", "basic.4y", "high.school"],
        "default": ["no", "no", "no", "no", "no", "no", "no", "no"],
        "housing": ["yes", "no", "yes", "no", "yes", "yes", "no", "no"],
        "loan": ["no", "yes", "no", "no", "no", "no", "no", "yes"],
        "contact": ["cellular", "cellular", "telephone", "cellular", "cellular", "cellular", "telephone", "cellular"],
        "month": ["may", "jun", "jul", "aug", "may", "jun", "jul", "aug"],
        "day_of_week": ["mon", "tue", "wed", "thu", "fri", "mon", "tue", "wed"],
        "campaign": [1, 2, 1, 3, 1, 2, 1, 4],
        "pdays": [999, 999, 6, 999, 999, 999, 999, 999],
        "previous": [0, 0, 1, 0, 0, 0, 0, 0],
        "poutcome": ["nonexistent", "nonexistent", "success", "nonexistent", "nonexistent", "nonexistent", "nonexistent", "nonexistent"],
        "emp.var.rate": [1.1, 1.1, -1.8, 1.4, 1.1, 1.1, -1.8, 1.4],
        "cons.price.idx": [93.994, 93.994, 92.893, 93.444, 93.994, 93.994, 92.893, 93.444],
        "cons.conf.idx": [-36.4, -36.4, -46.2, -36.1, -36.4, -36.4, -46.2, -36.1],
        "euribor3m": [4.857, 4.857, 1.281, 4.963, 4.857, 4.857, 1.281, 4.963],
        "nr.employed": [5191.0, 5191.0, 5099.1, 5228.1, 5191.0, 5191.0, 5099.1, 5228.1],
        "duration": [120, 300, 45, 210, 95, 400, 80, 150],
        "y": ["no", "no", "yes", "no", "no", "yes", "no", "no"],
    }
    df = pd.DataFrame(data)
    X, y, num_cols, cat_cols = prepare_features_and_target(
        df, include_economic_features=include_economic_features
    )
    pipeline = build_logistic_regression_pipeline(build_preprocessor(num_cols, cat_cols))
    pipeline.fit(X, y)
    return pipeline


def _payload_for(model: Pipeline) -> dict:
    """Construye un payload válido para el modelo de prueba."""
    payload = {
        "age": 40,
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "poutcome": "nonexistent",
    }
    for column in ECONOMIC_FEATURES:
        payload[column] = 1.0
    return {key: value for key, value in payload.items() if key in expected_feature_columns(model)}


# ==============================================================================
# 1. CARGA DEL ARTEFACTO
# ==============================================================================


def test_load_model_missing_raises_with_actionable_message(tmp_path: Path):
    """Un artefacto ausente debe explicar exactamente cómo generarlo."""
    missing = tmp_path / "best_model.joblib"

    with pytest.raises(ModelArtifactNotFound) as excinfo:
        load_model(missing)

    message = str(excinfo.value)
    assert "scripts/train.py" in message
    assert str(missing) in message


def test_load_model_corrupt_raises_model_artifact_error(tmp_path: Path):
    """Un archivo que no es un modelo serializado debe reportarse como error de artefacto."""
    corrupt = tmp_path / "best_model.joblib"
    corrupt.write_bytes(b"esto no es un modelo serializado")

    with pytest.raises(ModelArtifactError):
        load_model(corrupt)


def test_load_model_returns_pipeline(tmp_path: Path):
    """Un pipeline serializado se recupera sin pérdida."""
    model_path = tmp_path / "best_model.joblib"
    pipeline = _fit_sample_pipeline()
    joblib.dump(pipeline, model_path)

    loaded = load_model(model_path)

    assert isinstance(loaded, Pipeline)
    assert expected_feature_columns(loaded) == expected_feature_columns(pipeline)


# ==============================================================================
# 2. CONTRATO DE VARIABLES
# ==============================================================================


def test_expected_feature_columns_derives_from_pipeline():
    """Las columnas esperadas se derivan del preprocesador ajustado."""
    pipeline = _fit_sample_pipeline(include_economic_features=True)
    columns = expected_feature_columns(pipeline)

    for column in CLIENT_NUMERIC_FEATURES + CLIENT_CATEGORICAL_FEATURES + ECONOMIC_FEATURES:
        assert column in columns

    assert "duration" not in columns


def test_expected_feature_columns_without_economic_features():
    """El experimento sin variables económicas no las espera en la entrada."""
    pipeline = _fit_sample_pipeline(include_economic_features=False)

    assert uses_economic_features(pipeline) is False
    assert not set(expected_feature_columns(pipeline)) & set(ECONOMIC_FEATURES)


# ==============================================================================
# 3. CONSTRUCCIÓN DEL REGISTRO Y PREDICCIÓN
# ==============================================================================


def test_build_input_frame_returns_exactly_expected_columns():
    """El frame de entrada tiene exactamente las columnas del modelo, en orden."""
    pipeline = _fit_sample_pipeline()
    columns = expected_feature_columns(pipeline)
    payload = _payload_for(pipeline)
    payload["columna_extra"] = "valor ignorado"

    frame = build_input_frame(payload, columns)

    assert list(frame.columns) == columns
    assert len(frame) == 1
    assert "columna_extra" not in frame.columns


def test_build_input_frame_requires_every_expected_column():
    """Si falta una variable requerida se informa cuál falta."""
    pipeline = _fit_sample_pipeline()
    columns = expected_feature_columns(pipeline)
    payload = _payload_for(pipeline)
    payload.pop(columns[0])

    with pytest.raises(KeyError) as excinfo:
        build_input_frame(payload, columns)

    assert columns[0] in str(excinfo.value)


def test_predict_subscription_returns_probability_and_label():
    """La predicción devuelve probabilidad acotada y decisión coherente."""
    pipeline = _fit_sample_pipeline()
    payload = _payload_for(pipeline)

    result = predict_subscription(pipeline, payload)

    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in {0, 1}
    assert result["label"] in {"Si se suscribe", "No se suscribe"}
    assert result["expected_columns"] == expected_feature_columns(pipeline)

    expected_label = "Si se suscribe" if result["prediction"] == 1 else "No se suscribe"
    assert result["label"] == expected_label


def test_predict_subscription_handles_missing_probability():
    """Un clasificador sin `predict_proba` no debe romper la aplicación."""
    pipeline = _fit_sample_pipeline()
    payload = _payload_for(pipeline)

    class _SinProbabilidad:
        """Envuelve el pipeline y oculta el método de probabilidades."""

        def __init__(self, pipeline: Pipeline) -> None:
            self.named_steps = pipeline.named_steps

        def predict(self, frame: pd.DataFrame):
            return pipeline.predict(frame)

    result = predict_subscription(_SinProbabilidad(pipeline), payload)

    assert result["probability"] == 0.5
    assert result["prediction"] in {0, 1}


# ==============================================================================
# 4. METADATOS DEL DATASET
# ==============================================================================


def test_categorical_options_are_present_in_the_dataset():
    """Cada opción del formulario existe realmente en el dataset."""
    options = load_categorical_options(SAMPLE_DATA_PATH)
    df = pd.read_csv(SAMPLE_DATA_PATH, sep=";")

    for column in CLIENT_CATEGORICAL_FEATURES:
        observed = set(df[column].dropna().astype(str).unique().tolist())
        assert set(options[column]).issubset(observed)
        assert len(options[column]) > 1


def test_categorical_defaults_belong_to_the_options():
    """El valor por defecto del formulario es una opción válida."""
    options = load_categorical_options(SAMPLE_DATA_PATH)
    defaults = load_categorical_defaults(SAMPLE_DATA_PATH)

    for column in CLIENT_CATEGORICAL_FEATURES:
        assert defaults[column] in options[column]


def test_numeric_ranges_bracket_observed_values():
    """El rango numérico propuesto contiene a los valores observados."""
    ranges = load_numeric_ranges(SAMPLE_DATA_PATH)
    df = pd.read_csv(SAMPLE_DATA_PATH, sep=";")

    for column in CLIENT_NUMERIC_FEATURES + ECONOMIC_FEATURES:
        minimum, maximum, median = ranges[column]
        observed = df[column].dropna()

        assert minimum <= median <= maximum
        assert observed.min() >= minimum
        assert observed.max() <= maximum


def test_load_dataset_metadata_returns_all_sections():
    """Los metadatos exponen las tres secciones que consume el formulario."""
    metadata = load_dataset_metadata(SAMPLE_DATA_PATH)

    assert set(metadata) == {
        "categorical_options",
        "categorical_defaults",
        "numeric_ranges",
    }
    assert set(metadata["categorical_options"]) == set(CLIENT_CATEGORICAL_FEATURES)
    assert set(metadata["numeric_ranges"]) == set(
        CLIENT_NUMERIC_FEATURES + ECONOMIC_FEATURES
    )


# ==============================================================================
# 5. MÉTRICAS
# ==============================================================================


def test_load_metrics_returns_empty_when_file_missing(tmp_path: Path):
    """Sin archivo de métricas la aplicación debe poder seguir funcionando."""
    assert load_metrics(tmp_path / "metrics.json") == {}


def test_summarize_metrics_builds_long_table():
    """El resumen aplana la estructura anidada de `metrics.json`."""
    metrics = {
        "sin_variables_economicas": {
            "dummy_baseline": {"f1_macro": 0.47, "accuracy": 0.88},
            "random_forest": {"f1_macro": 0.71, "accuracy": 0.85},
        },
        "con_variables_economicas": {
            "logistic_regression": {"f1_macro": 0.68, "accuracy": 0.86},
        },
    }

    summary = summarize_metrics(metrics)

    assert len(summary) == 3
    assert set(summary.columns) >= {"experimento", "modelo", "f1_macro"}
    assert summary.iloc[0]["f1_macro"] == max(summary["f1_macro"])


def test_summarize_metrics_handles_empty_input():
    """Sin métricas se devuelve un DataFrame vacío, no un error."""
    assert summarize_metrics({}).empty


# ==============================================================================
# 6. RUTAS DEL PROYECTO
# ==============================================================================


def test_default_paths_point_to_the_repository():
    """Las rutas por defecto apuntan a los artefactos del repositorio."""
    assert DEFAULT_DATA_PATH.exists()
    assert DEFAULT_DATA_PATH.name == "bank-additional.csv"
    assert REPO_ROOT.name in str(DEFAULT_DATA_PATH)
