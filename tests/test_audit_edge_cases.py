"""Validación y trazabilidad de la auditoría sin utilizar el dataset completo."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uasd_ds_2_final_project import audit


def test_prepare_dataset_rejects_a_missing_target():
    """La limpieza exige la variable objetivo antes de procesar columnas."""
    with pytest.raises(ValueError, match="Falta la columna objetivo obligatoria: y"):
        audit.prepare_dataset(pd.DataFrame({"age": [30]}))


def test_prepare_dataset_rejects_null_target_values():
    """Una etiqueta nula no puede convertirse silenciosamente en la clase negativa."""
    with pytest.raises(ValueError, match="Etiquetas no válidas"):
        audit.prepare_dataset(pd.DataFrame({"y": ["yes", None]}))


def test_prepare_dataset_preserves_duplicates_and_does_not_mutate_source():
    """La limpieza conserva observaciones duplicadas, índices y duración original."""
    source = pd.DataFrame(
        {"age": [30, 30], "job": [" admin. "] * 2, "duration": [60, 60], "y": [" YES "] * 2},
        index=[7, 8],
    )
    original = source.copy(deep=True)

    prepared = audit.prepare_dataset(source)

    pd.testing.assert_frame_equal(source, original)
    pd.testing.assert_index_equal(prepared.index, source.index)
    assert list(prepared.columns) == list(source.columns)
    assert prepared.duplicated().sum() == 1
    assert prepared["job"].tolist() == ["admin.", "admin."]
    assert prepared["y"].tolist() == [1, 1]
    assert prepared["y"].dtype == np.dtype("int8")
    pd.testing.assert_series_equal(prepared["duration"], source["duration"])


def test_audit_dataset_reports_quality_and_provenance_without_altering_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """La auditoría cuenta anomalías y conserva la huella verificable de la fuente."""
    source = pd.DataFrame(
        {
            "age": [10.0, 10.0, 10.0, 10.0, 100.0, np.nan, np.inf, -np.inf],
            "job": [" unknown ", " unknown ", "admin.", "admin.", None, "UNKNOWN", "retired", "retired"],
            "y": ["no", "no", "yes", "no", "yes", "no", "yes", "no"],
        }
    )
    original = source.copy(deep=True)
    source_bytes = source.to_csv(sep=";", index=False).encode("utf-8")
    source_path = tmp_path / "bank.csv"
    source_path.write_bytes(source_bytes)
    monkeypatch.setattr(audit, "ROOT_DIR", tmp_path)

    report = audit.audit_dataset(source, source_path)

    assert report["source_file"] == "bank.csv"
    assert report["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert report["source_shape"] == {"rows": 8, "columns": 3}
    assert report["columns"] == ["age", "job", "y"]
    assert report["missing_values_by_column"] == {"age": 1, "job": 1, "y": 0}
    assert report["missing_values_total"] == 2
    assert report["non_finite_numeric_by_column"] == {"age": 3}
    assert report["exact_duplicate_rows"] == 1
    assert report["unknown_category_counts"] == {"job": 3, "y": 0}
    assert report["target_counts"] == {"no": 5, "yes": 3}
    assert report["target_percentages"] == {"no": 62.5, "yes": 37.5}
    assert report["numeric_summary_min_median_max"] == {
        "age": {"min": 10.0, "median": 10.0, "max": 100.0}
    }
    assert report["potential_iqr_outliers"]["counts_by_column"] == {"age": 1}
    pd.testing.assert_frame_equal(source, original)
    assert source_path.read_bytes() == source_bytes
