"""
Pruebas Unitarias para el Módulo de Modelos
===========================================
Verifica que las fábricas de modelos construyan Pipelines válidos
y que estos cumplan con el contrato de ajuste (.fit) y predicción (.predict).
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from uasd_ds_2_final_project.data import (
    build_preprocessor,
    prepare_features_and_target,
)
from uasd_ds_2_final_project.models import get_model_factories


def _create_sample_dataset() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Genera datos sintéticos mínimos para entrenar y evaluar pipelines."""
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
    return prepare_features_and_target(df, include_economic_features=True)


def test_factories_return_valid_pipelines():
    """Verifica que cada fábrica devuelva un objeto Pipeline con preprocessor y classifier."""
    X, y, num_cols, cat_cols = _create_sample_dataset()
    factories = get_model_factories()

    assert "dummy_baseline" in factories
    assert "logistic_regression" in factories
    assert "random_forest" in factories

    for name, factory in factories.items():
        preprocessor = build_preprocessor(num_cols, cat_cols)
        pipeline = factory(preprocessor)

        assert isinstance(pipeline, Pipeline), f"{name} no es una instancia de Pipeline"
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps


def test_pipeline_fit_and_predict_contract():
    """Verifica que todos los modelos completen el ciclo de ajuste y predicción."""
    X, y, num_cols, cat_cols = _create_sample_dataset()
    factories = get_model_factories()

    for name, factory in factories.items():
        preprocessor = build_preprocessor(num_cols, cat_cols)
        pipeline = factory(preprocessor)

        # Ajuste (fit)
        pipeline.fit(X, y)

        # Predicción (predict)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})
