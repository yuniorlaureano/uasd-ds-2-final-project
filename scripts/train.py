"""
Script de Orquestación y Entrenamiento de Modelos
=================================================
Este script ejecuta el pipeline completo de entrenamiento y evaluación.

Responde de forma cuantitativa a la pregunta de investigación:
"¿En qué medida la incorporación de variables económicas mejora la predicción
de respuesta a campañas bancarias frente a modelos basados únicamente en
información del cliente y contactos anteriores?"

Artefactos generados:
- reports/metrics.json: Métricas de rendimiento de todos los modelos y experimentos.
- reports/confusion_matrix.png: Matriz de confusión del mejor modelo.
- reports/roc_curves.png: Comparación de curvas ROC.
- models/best_model.joblib: Pipeline completo del modelo con mejor desempeño.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from uasd_ds_2_final_project.data import (
    build_preprocessor,
    load_raw_data,
    prepare_features_and_target,
    split_data,
)
from uasd_ds_2_final_project.models import get_model_factories

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Configura los argumentos por línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento para predicción de campañas bancarias."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "bank-additional-full.csv"),
        help="Ruta al archivo CSV con los datos crudos.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción del conjunto de prueba (por defecto: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )
    return parser.parse_args()


def evaluate_model(
    pipeline,
    X_test,
    y_test,
    train_seconds: float,
) -> dict[str, float]:
    """Calcula las métricas clave de evaluación en el conjunto de prueba."""
    start_eval = perf_counter()
    y_pred = pipeline.predict(X_test)
    eval_seconds = perf_counter() - start_eval

    # Intentar obtener probabilidades para ROC-AUC (DummyClassifier con most_frequent no siempre tiene)
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
    except (AttributeError, IndexError):
        auc = 0.5

    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_pos = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_target_yes": round(f1_pos, 4),
        "recall_target_yes": round(rec, 4),
        "precision_target_yes": round(prec, 4),
        "roc_auc": round(auc, 4),
        "train_seconds": round(train_seconds, 4),
        "inference_ms_per_sample": round((eval_seconds / len(X_test)) * 1000, 4),
    }


def main() -> None:
    args = parse_arguments()

    reports_dir = ROOT_DIR / "reports"
    models_dir = ROOT_DIR / "models"
    reports_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("INICIANDO PIPELINE DE ENTRENAMIENTO - BANK MARKETING PREDICTION")
    print("=" * 80)
    print(f"Cargando datos desde: {args.data_path}")

    df_raw = load_raw_data(args.data_path)
    print(f"Dimensiones del dataset cargado: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas.")

    # Experimentos para responder a la pregunta de investigación:
    # 1. 'sin_economicas': Solo información del cliente y contactos previos.
    # 2. 'con_economicas': Cliente + indicadores socioeconómicos (Banco de Portugal).
    experiments = [
        {"name": "sin_variables_economicas", "include_economic": False},
        {"name": "con_variables_economicas", "include_economic": True},
    ]

    model_factories = get_model_factories()
    all_metrics: dict[str, dict] = {}
    trained_pipelines: dict[str, tuple] = {}

    best_score = -1.0
    best_model_key = ""
    best_pipeline = None
    best_test_data = None

    for exp in experiments:
        exp_name = exp["name"]
        include_econ = exp["include_economic"]

        print("\n" + "-" * 80)
        print(f"EJECUTANDO EXPERIMENTO: {exp_name.upper()}")
        print("-" * 80)

        # 1. Preparar características y variable objetivo (excluyendo 'duration')
        X, y, num_cols, cat_cols = prepare_features_and_target(
            df_raw,
            include_economic_features=include_econ,
            drop_leakage=True,
        )

        # 2. División estratificada
        X_train, X_test, y_train, y_test = split_data(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        all_metrics[exp_name] = {}

        for model_name, factory in model_factories.items():
            run_key = f"{model_name}__{exp_name}"
            print(f"-> Entrenando [{model_name}] ...", end=" ", flush=True)

            # Construir preprocesador con las columnas activas de este experimento
            preprocessor = build_preprocessor(num_cols, cat_cols)
            pipeline = factory(preprocessor)

            # Entrenar y medir tiempo
            t0 = perf_counter()
            pipeline.fit(X_train, y_train)
            train_time = perf_counter() - t0

            # Evaluar métricas
            metrics = evaluate_model(pipeline, X_test, y_test, train_time)
            all_metrics[exp_name][model_name] = metrics
            trained_pipelines[run_key] = (pipeline, X_test, y_test)

            print(f"Listo ({train_time:.2f}s) | F1-Macro: {metrics['f1_macro']} | ROC-AUC: {metrics['roc_auc']}")

            # Seleccionar el mejor modelo según F1-Macro (métrica clave para desbalance)
            if metrics["f1_macro"] > best_score:
                best_score = metrics["f1_macro"]
                best_model_key = run_key
                best_pipeline = pipeline
                best_test_data = (X_test, y_test)

    # ==============================================================================
    # GUARDAR ARTEFACTOS Y REPORTES
    # ==============================================================================
    print("\n" + "=" * 80)
    print("GUARDANDO ARTEFACTOS Y GENERANDO REPORTES")
    print("=" * 80)

    # 1. Guardar métricas en JSON
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"[OK] Metricas exportadas a: {metrics_path}")

    # 2. Guardar el mejor modelo serializado
    if best_pipeline is not None:
        model_artifact_path = models_dir / "best_model.joblib"
        joblib.dump(best_pipeline, model_artifact_path)
        print(f"[OK] Mejor modelo guardado en: {model_artifact_path} (Ganador: {best_model_key})")

        # 3. Generar Matriz de Confusión del mejor modelo
        X_test_best, y_test_best = best_test_data
        y_pred_best = best_pipeline.predict(X_test_best)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test_best,
            y_pred_best,
            display_labels=["No Suscribe (0)", "Suscribe (1)"],
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"Matriz de Confusión - {best_model_key}")
        fig.tight_layout()
        fig_path = reports_dir / "confusion_matrix.png"
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        print(f"[OK] Matriz de confusion guardada en: {fig_path}")

        # 4. Generar Curvas ROC comparativas para la condición con variables económicas
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        for model_name in model_factories.keys():
            key = f"{model_name}__con_variables_economicas"
            if key in trained_pipelines:
                pipe, xt, yt = trained_pipelines[key]
                if hasattr(pipe.named_steps["classifier"], "predict_proba"):
                    RocCurveDisplay.from_estimator(pipe, xt, yt, name=model_name, ax=ax_roc)
        ax_roc.plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.50)")
        ax_roc.set_title("Comparación de Curvas ROC (Con Variables Económicas)")
        ax_roc.legend(loc="lower right")
        fig_roc.tight_layout()
        roc_path = reports_dir / "roc_curves.png"
        fig_roc.savefig(roc_path, dpi=180)
        plt.close(fig_roc)
        print(f"[OK] Curvas ROC guardadas en: {roc_path}")

        print("\nReporte de Clasificación del Mejor Modelo:")
        print(classification_report(y_test_best, y_pred_best, target_names=["No (0)", "Yes (1)"], digits=4))

    print("=" * 80)
    print("PIPELINE DE ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()
