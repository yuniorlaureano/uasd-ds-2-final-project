"""Contratos de artefactos y métricas con archivos temporales independientes."""

import json
from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from uasd_ds_2_final_project.inference import (
    ModelArtifactError,
    expected_feature_columns,
    load_metrics,
    load_model,
)


def test_load_model_rejects_a_serialized_object_that_is_not_a_pipeline(tmp_path: Path):
    """Un archivo joblib válido también debe satisfacer el contrato del modelo."""
    model_path = tmp_path / "wrong_model.joblib"
    joblib.dump({"classifier": "not a pipeline"}, model_path)

    with pytest.raises(ModelArtifactError, match="no es un Pipeline"):
        load_model(model_path)


def test_expected_feature_columns_rejects_a_pipeline_without_preprocessor():
    """Sin metadatos de entrada no se permite adivinar las variables del cliente."""
    model = Pipeline([("classifier", DummyClassifier())])

    with pytest.raises(ModelArtifactError, match="feature_names_in_"):
        expected_feature_columns(model)


def test_load_metrics_preserves_valid_nested_json(tmp_path: Path):
    """Un reporte válido conserva sus experimentos, números y texto UTF-8."""
    metrics_path = tmp_path / "metrics.json"
    expected = {
        "con_variables_economicas": {
            "random_forest": {"f1_macro": 0.71, "accuracy": 0.86, "nota": "evaluación"}
        }
    }
    metrics_path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")

    assert load_metrics(str(metrics_path)) == expected


def test_load_metrics_returns_empty_for_malformed_json(tmp_path: Path):
    """Un reporte truncado no impide usar la aplicación de predicción."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"con_variables_economicas":', encoding="utf-8")

    assert load_metrics(metrics_path) == {}


def test_load_metrics_returns_empty_for_json_with_a_non_object_root(tmp_path: Path):
    """JSON sintácticamente válido con raíz de tipo lista se descarta."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('[{"f1_macro": 0.71}]', encoding="utf-8")

    assert load_metrics(metrics_path) == {}
