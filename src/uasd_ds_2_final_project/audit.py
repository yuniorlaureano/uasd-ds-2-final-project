"""Audit and prepare the Bank Marketing dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT_DIR / "data" / "raw" / "bank-additional-full.csv"
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "bank-additional-clean.csv"
REPORT_PATH = ROOT_DIR / "reports" / "data_audit.json"
TARGET_COLUMN = "y"


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Trim text fields and encode the target, preserving all source rows."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Falta la columna objetivo obligatoria: {TARGET_COLUMN}")

    prepared = df.copy()
    text_columns = prepared.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        prepared[column] = prepared[column].str.strip()

    target_labels = prepared[TARGET_COLUMN].astype("string").str.lower()
    invalid_labels = sorted(set(target_labels.dropna().unique()) - {"yes", "no"})
    if target_labels.isna().any() or invalid_labels:
        raise ValueError(f"Etiquetas no válidas en '{TARGET_COLUMN}': {invalid_labels}")

    prepared[TARGET_COLUMN] = target_labels.map({"no": 0, "yes": 1}).astype("int8")
    return prepared


def audit_dataset(df: pd.DataFrame, source_path: Path) -> dict:
    """Build reproducible quality checks without silently altering observations."""
    numeric = df.select_dtypes(include="number")
    non_finite = {
        column: int((~np.isfinite(numeric[column].to_numpy(dtype=float))).sum())
        for column in numeric.columns
    }

    numeric_summary = {}
    outlier_counts = {}
    for column in numeric.columns:
        values = numeric[column].replace([np.inf, -np.inf], np.nan).dropna()
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        outlier_counts[column] = int(
            ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()
        )
        numeric_summary[column] = {
            "min": float(values.min()),
            "median": float(values.median()),
            "max": float(values.max()),
        }

    categorical = df.select_dtypes(include=["object", "string"])
    unknown_counts = {
        column: int(
            categorical[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("unknown")
            .sum()
        )
        for column in categorical.columns
    }
    target_counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()

    return {
        "source_file": source_path.relative_to(ROOT_DIR).as_posix(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values_by_column": {
            column: int(count) for column, count in df.isna().sum().items()
        },
        "missing_values_total": int(df.isna().sum().sum()),
        "non_finite_numeric_by_column": non_finite,
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_handling": (
            "Preserved: without a record identifier, exact matches cannot be "
            "confirmed as erroneous duplicate observations."
        ),
        "unknown_category_counts": unknown_counts,
        "target_counts": {str(label): int(count) for label, count in target_counts.items()},
        "target_percentages": {
            str(label): round(float(count / len(df) * 100), 4)
            for label, count in target_counts.items()
        },
        "numeric_summary_min_median_max": numeric_summary,
        "potential_iqr_outliers": {
            "method": "Outside Q1 - 1.5*IQR or Q3 + 1.5*IQR; flagged, not removed.",
            "counts_by_column": outlier_counts,
        },
    }


def main() -> None:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el archivo fuente: {SOURCE_PATH}")

    raw_df = pd.read_csv(SOURCE_PATH, sep=";")
    report = audit_dataset(raw_df, SOURCE_PATH)
    processed_df = prepare_dataset(raw_df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_PATH, sep=";", index=False, encoding="utf-8")

    report["processing"] = {
        "output_file": PROCESSED_PATH.relative_to(ROOT_DIR).as_posix(),
        "output_shape": {
            "rows": int(processed_df.shape[0]),
            "columns": int(processed_df.shape[1]),
        },
        "target_encoding": {"no": 0, "yes": 1},
        "text_processing": "Leading and trailing whitespace removed.",
        "rows_removed": 0,
        "duration": (
            "Retained in the processed file for traceability; excluded by the "
            "model feature-preparation pipeline to prevent target leakage."
        ),
        "unknown_categories": "Preserved as explicit category values.",
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Fuente: {SOURCE_PATH.relative_to(ROOT_DIR)}")
    print(f"Dimensiones: {raw_df.shape[0]} filas x {raw_df.shape[1]} columnas")
    print(f"Valores faltantes: {report['missing_values_total']}")
    print(f"Filas exactamente duplicadas (conservadas): {report['exact_duplicate_rows']}")
    print(f"Dataset procesado: {PROCESSED_PATH.relative_to(ROOT_DIR)}")
    print(f"Reporte de auditoría: {REPORT_PATH.relative_to(ROOT_DIR)}")
