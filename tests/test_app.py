"""
Pruebas de Humo de la Aplicación Streamlit
==========================================
Verifica que el script de la interfaz se ejecute sin excepciones y que el flujo
principal (captura del cliente y predicción) funcione.

Estas pruebas se ejecutan con `streamlit.testing.v1.AppTest`, que corre el script
en un entorno headless sin necesidad de levantar un servidor.

Si el modelo entrenado (`models/best_model.joblib`) todavía no existe, las pruebas
que dependen de él se omiten: en ese escenario la aplicación debe explicar cómo
generarlo, y eso también se verifica.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from uasd_ds_2_final_project.inference import DEFAULT_MODEL_PATH

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "uasd_ds_2_final_project" / "app.py"


def _run_app() -> AppTest:
    """Ejecuta el script de la aplicación en modo headless."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.run()
    return app


def test_app_renders_without_exceptions():
    """La página se construye completa y sin errores."""
    app = _run_app()

    assert not app.exception, [str(error) for error in app.exception]
    assert app.title[0].value.startswith("Predicción de suscripciones")
    assert len(app.tabs) == 3


def test_app_explains_how_to_generate_the_model_when_artifact_is_missing():
    """Sin artefacto entrenado la app indica el comando exacto, sin romperse."""
    app = _run_app()

    assert not app.exception, [str(error) for error in app.exception]

    if DEFAULT_MODEL_PATH.exists():
        pytest.skip("El modelo ya está entrenado en este entorno.")

    warnings = " ".join(warning.value for warning in app.warning)
    assert "scripts/train.py" in warnings
    assert any("Entrenar" in button.label for button in app.button)


def test_app_shows_performance_tab_with_metrics():
    """La pestaña de desempeño expone la tabla de métricas y los gráficos."""
    app = _run_app()

    assert not app.exception, [str(error) for error in app.exception]

    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("Requiere los artefactos generados por scripts/train.py.")

    # Una tabla de métricas y dos tablas de la ficha del modelo.
    assert len(app.dataframe) >= 3
    assert len(app.image) == 2


def test_prediction_flow_returns_probability_and_decision():
    """Enviar el formulario produce una probabilidad y una decisión."""
    app = _run_app()

    assert not app.exception, [str(error) for error in app.exception]

    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("Requiere el modelo entrenado por scripts/train.py.")

    submit_buttons = [button for button in app.button if "Predecir" in button.label]
    assert submit_buttons, "No se encontró el botón de predicción."

    submit_buttons[0].click()
    app.run()

    assert not app.exception, [str(error) for error in app.exception]
    assert app.metric, "La predicción no mostró la métrica de probabilidad."
    assert app.metric[0].value.endswith("%")

    decisions = " ".join([item.value for item in app.success] + [item.value for item in app.warning])
    assert "se suscribe" in decisions
