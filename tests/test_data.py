"""
Pruebas Unitarias para el Módulo de Datos
=========================================
Verifica la integridad de las funciones de carga, validación,
prevención de fuga de información (data leakage) y preprocesamiento.
"""

from __future__ import annotations

import pandas as pd
import pytest

from uasd_ds_2_final_project.data import (
    CLIENT_CATEGORICAL_FEATURES,
    CLIENT_NUMERIC_FEATURES,
    ECONOMIC_FEATURES,
    build_preprocessor,
    prepare_features_and_target,
    split_data,
    validate_data,
)


def _create_sample_df() -> pd.DataFrame:
    """Crea un DataFrame sintético con todas las columnas obligatorias para pruebas."""
    return pd.DataFrame({
        "age": [30, 40, 50, 60],
        "job": ["admin.", "technician", "services", "blue-collar"],
        "marital": ["married", "single", "divorced", "married"],
        "education": ["university.degree", "high.school", "basic.9y", "professional.course"],
        "default": ["no", "no", "unknown", "no"],
        "housing": ["yes", "no", "yes", "no"],
        "loan": ["no", "yes", "no", "no"],
        "contact": ["cellular", "cellular", "telephone", "cellular"],
        "month": ["may", "jun", "jul", "aug"],
        "day_of_week": ["mon", "tue", "wed", "thu"],
        "campaign": [1, 2, 1, 3],
        "pdays": [999, 999, 6, 999],
        "previous": [0, 0, 1, 0],
        "poutcome": ["nonexistent", "nonexistent", "success", "nonexistent"],
        "emp.var.rate": [1.1, 1.1, -1.8, 1.4],
        "cons.price.idx": [93.994, 93.994, 92.893, 93.444],
        "cons.conf.idx": [-36.4, -36.4, -46.2, -36.1],
        "euribor3m": [4.857, 4.857, 1.281, 4.963],
        "nr.employed": [5191.0, 5191.0, 5099.1, 5228.1],
        "duration": [261, 149, 226, 151],  # Columna con fuga de datos
        "y": ["no", "no", "yes", "no"],   # Variable objetivo
    })


def test_validate_data_rejects_empty():
    """Valida que se rechace un DataFrame vacío."""
    with pytest.raises(ValueError, match="está vacío"):
        validate_data(pd.DataFrame())


def test_validate_data_rejects_missing_target():
    """Valida que se rechace un DataFrame si falta la variable objetivo 'y'."""
    df = _create_sample_df().drop(columns=["y"])
    with pytest.raises(ValueError, match="Falta la columna objetivo"):
        validate_data(df)


def test_prepare_features_strictly_excludes_duration_leakage():
    """
    CRÍTICO: Verifica que 'duration' NUNCA esté presente en el conjunto X
    para evitar fuga de información temporal.
    """
    df = _create_sample_df()
    X, _, _, _ = prepare_features_and_target(df, drop_leakage=True)
    assert "duration" not in X.columns


def test_prepare_features_economic_ablation_switch():
    """
    Verifica que el interruptor de variables económicas funcione:
    - Con False: solo datos del cliente.
    - Con True: incluye variables macroeconómicas.
    """
    df = _create_sample_df()

    # Experimento sin variables macroeconómicas
    X_no_econ, _, num_no_econ, _ = prepare_features_and_target(df, include_economic_features=False)
    for econ_col in ECONOMIC_FEATURES:
        assert econ_col not in X_no_econ.columns
        assert econ_col not in num_no_econ

    # Experimento con variables macroeconómicas
    X_econ, _, num_econ, _ = prepare_features_and_target(df, include_economic_features=True)
    for econ_col in ECONOMIC_FEATURES:
        assert econ_col in X_econ.columns
        assert econ_col in num_econ


def test_target_is_binary_numeric():
    """Verifica que la variable 'y' se convierta correctamente a valores binarios 0 y 1."""
    df = _create_sample_df()
    _, y, _, _ = prepare_features_and_target(df)
    assert set(y.unique()).issubset({0, 1})
    assert y.iloc[2] == 1  # 'yes' -> 1
    assert y.iloc[0] == 0  # 'no' -> 0


def test_split_data_preserves_total_rows():
    """Verifica que la división train/test no pierda registros."""
    df = _create_sample_df()
    X, y, _, _ = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5, random_state=42)

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_build_preprocessor_transforms_data():
    """Verifica que el ColumnTransformer escale numéricas y codifique categóricas."""
    df = _create_sample_df()
    X, _, num_cols, cat_cols = prepare_features_and_target(df)

    preprocessor = build_preprocessor(num_cols, cat_cols)
    transformed = preprocessor.fit_transform(X)

    # Debe ser una matriz numérica de numpy de 2 dimensiones
    assert transformed.ndim == 2
    assert transformed.shape[0] == len(df)
