"""Casos unitarios de carga, validación y preprocesamiento con datos sintéticos."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uasd_ds_2_final_project.data import (
    build_preprocessor,
    load_raw_data,
    prepare_features_and_target,
    split_data,
    validate_data,
)


@pytest.fixture
def bank_data() -> pd.DataFrame:
    """Cuatro clientes independientes de los archivos y artefactos del proyecto."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "campaign": [1, 2, 1, 3],
            "pdays": [999, 6, 999, 3],
            "previous": [0, 1, 0, 2],
            "job": ["admin.", "technician", "services", "retired"],
            "marital": ["single", "married", "divorced", "married"],
            "education": ["high.school", "university.degree", "basic.9y", "basic.4y"],
            "default": ["no", "no", "unknown", "no"],
            "housing": ["yes", "no", "yes", "no"],
            "loan": ["no", "yes", "no", "no"],
            "contact": ["cellular", "telephone", "cellular", "telephone"],
            "month": ["may", "jun", "jul", "aug"],
            "day_of_week": ["mon", "tue", "wed", "thu"],
            "poutcome": ["nonexistent", "success", "nonexistent", "failure"],
            "emp.var.rate": [1.1, -1.8, 1.4, -0.1],
            "cons.price.idx": [93.994, 92.893, 93.444, 93.2],
            "cons.conf.idx": [-36.4, -46.2, -36.1, -40.0],
            "euribor3m": [4.857, 1.281, 4.963, 2.0],
            "nr.employed": [5191.0, 5099.1, 5228.1, 5100.0],
            "duration": [60, 120, 90, 180],
            "y": ["no", "yes", "no", "yes"],
        },
        index=pd.Index([10, 20, 30, 40], name="customer"),
    )


def test_load_raw_data_reports_the_missing_path(tmp_path: Path):
    """Un CSV inexistente informa la ruta que debe corregirse."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError) as excinfo:
        load_raw_data(missing_path)

    assert str(missing_path) in str(excinfo.value)


def test_load_raw_data_reads_semicolons_and_quoted_fields(tmp_path: Path):
    """El delimitador europeo respeta los puntos y coma dentro de comillas."""
    source_path = tmp_path / "bank.csv"
    source_path.write_text('age;job;y\n30;"admin.; senior";yes\n', encoding="utf-8")

    loaded = load_raw_data(str(source_path))

    pd.testing.assert_frame_equal(
        loaded,
        pd.DataFrame({"age": [30], "job": ["admin.; senior"], "y": ["yes"]}),
    )


def test_validate_data_identifies_missing_predictors(bank_data: pd.DataFrame):
    """La validación identifica cada predictor ausente antes de entrenar."""
    incomplete = bank_data.drop(columns=["age", "job", "euribor3m"])

    with pytest.raises(ValueError, match="Faltan columnas obligatorias") as excinfo:
        validate_data(incomplete)

    for column in ("age", "job", "euribor3m"):
        assert column in str(excinfo.value)


def test_prepare_features_normalizes_labels_without_mutating_source(bank_data: pd.DataFrame):
    """Mayúsculas, espacios y etiquetas binarias se normalizan sobre una copia."""
    bank_data["y"] = [" YES ", " No ", "1", " 0 "]
    original = bank_data.copy(deep=True)

    features, target, _, _ = prepare_features_and_target(bank_data)

    pd.testing.assert_series_equal(
        target,
        pd.Series([1, 0, 1, 0], index=bank_data.index, name="y", dtype=int),
    )
    features.loc[10, "age"] = 99
    pd.testing.assert_frame_equal(bank_data, original)


def test_split_data_keeps_class_balance_and_aligned_disjoint_partitions():
    """La partición estratificada conserva la clase minoritaria y sus índices."""
    features = pd.DataFrame({"age": np.arange(100)}, index=np.arange(1000, 1100))
    target = pd.Series([0] * 80 + [1] * 20, index=features.index, name="y")

    train_x, test_x, train_y, test_y = split_data(
        features, target, test_size=0.25, random_state=17
    )

    assert train_y.value_counts().to_dict() == {0: 60, 1: 15}
    assert test_y.value_counts().to_dict() == {0: 20, 1: 5}
    assert set(train_x.index).isdisjoint(test_x.index)
    assert set(train_x.index) | set(test_x.index) == set(features.index)
    pd.testing.assert_frame_equal(train_x, features.loc[train_y.index])
    pd.testing.assert_frame_equal(test_x, features.loc[test_y.index])
    pd.testing.assert_series_equal(train_y, target.loc[train_x.index])
    pd.testing.assert_series_equal(test_y, target.loc[test_x.index])


def test_preprocessor_accepts_a_category_absent_from_training():
    """Una ocupación nueva produce ceros en su bloque categórico sin fallar."""
    preprocessor = build_preprocessor(["age"], ["job"])
    preprocessor.fit(pd.DataFrame({"age": [20, 40], "job": ["admin.", "technician"]}))

    transformed = preprocessor.transform(pd.DataFrame({"age": [30], "job": ["retired"]}))

    np.testing.assert_allclose(transformed, [[0.0, 0.0, 0.0]])


def test_preprocessor_scales_new_rows_using_only_training_statistics():
    """Inferencia usa la media 30 y desviación 10 aprendidas en entrenamiento."""
    preprocessor = build_preprocessor(["age"], [])
    preprocessor.fit(pd.DataFrame({"age": [20, 40]}))

    transformed = preprocessor.transform(pd.DataFrame({"age": [60, 80]}))

    np.testing.assert_allclose(transformed, [[3.0], [5.0]])
