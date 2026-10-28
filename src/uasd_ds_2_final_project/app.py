"""
Aplicación Web: Predicción de Suscripciones a Depósitos a Plazo
================================================================
Interfaz de usuario (Streamlit) sobre el pipeline de entrenamiento del proyecto.

Permite:
1. Capturar el perfil de un cliente potencial.
2. Estimar en vivo la probabilidad de que suscriba un depósito a plazo.
3. Consultar el desempeño del modelo entrenado (métricas, matriz de confusión y ROC).
4. Consultar la ficha del modelo (variables, supuestos y limitaciones).

La lógica de negocio vive en `uasd_ds_2_final_project.inference`; este módulo
se limita a la presentación.

Ejecución:
    uv run streamlit run src/uasd_ds_2_final_project/app.py
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import streamlit as st

from uasd_ds_2_final_project.data import (
    CLIENT_CATEGORICAL_FEATURES,
    CLIENT_NUMERIC_FEATURES,
    ECONOMIC_FEATURES,
)
from uasd_ds_2_final_project.inference import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORTS_DIR,
    INTEGER_FEATURES,
    TRAIN_SCRIPT_PATH,
    ModelArtifactError,
    ModelArtifactNotFound,
    load_dataset_metadata,
    load_metrics,
    load_model,
    predict_subscription,
    summarize_metrics,
    uses_economic_features,
)

# ==============================================================================
# 1. TEXTOS DE LA INTERFAZ
# ==============================================================================

VARIABLE_LABELS: dict[str, str] = {
    "age": "Edad",
    "campaign": "Contactos en esta campaña",
    "pdays": "Días desde el último contacto",
    "previous": "Contactos en campañas anteriores",
    "job": "Ocupación",
    "marital": "Estado civil",
    "education": "Nivel educativo",
    "default": "Tiene crédito en mora",
    "housing": "Tiene préstamo hipotecario",
    "loan": "Tiene préstamo personal",
    "contact": "Tipo de contacto",
    "month": "Mes del último contacto",
    "day_of_week": "Día de la semana del último contacto",
    "poutcome": "Resultado de campañas previas",
    "emp.var.rate": "Tasa de variación del empleo",
    "cons.price.idx": "Índice de precios al consumidor",
    "cons.conf.idx": "Índice de confianza del consumidor",
    "euribor3m": "Euribor a 3 meses",
    "nr.employed": "Empleados (miles)",
}

VARIABLE_HELP: dict[str, str] = {
    "pdays": "999 significa que el cliente nunca fue contactado en una campaña anterior.",
    "campaign": "Número de contactos realizados durante la campaña actual.",
    "previous": "Número de contactos realizados en campañas anteriores.",
    "default": "Valores posibles: no, yes, unknown.",
    "poutcome": "Resultado de la campaña previa: failure, nonexistent o success.",
    "emp.var.rate": "Indicador trimestral de variación del empleo.",
    "nr.employed": "Número de empleados, en miles, con datos trimestrales.",
}

CLIENT_INFO = {
    "age": "Edad del cliente al momento del contacto.",
    "campaign": "Contactos realizados durante la campaña vigente.",
    "pdays": "Días transcurridos desde el último contacto (999 = nunca contactado).",
    "previous": "Cantidad de contactos en campañas anteriores.",
    "job": "Ocupación declarada por el cliente.",
    "marital": "Estado civil declarado.",
    "education": "Nivel educativo declarado.",
    "default": "Indica si el cliente tiene créditos en mora.",
    "housing": "Indica si el cliente tiene préstamo hipotecario.",
    "loan": "Indica si el cliente tiene préstamo personal.",
    "contact": "Canal de comunicación utilizado (telefónico o móvil).",
    "month": "Mes del último contacto.",
    "day_of_week": "Día de la semana del último contacto.",
    "poutcome": "Resultado de la campaña de marketing anterior.",
}

ECONOMIC_INFO = {
    "emp.var.rate": "Tasa de variación del empleo (Banco de Portugal).",
    "cons.price.idx": "Índice de precios al consumidor.",
    "cons.conf.idx": "Índice de confianza del consumidor.",
    "euribor3m": "Tasa Euribor a 3 meses.",
    "nr.employed": "Número de empleados (miles), dato trimestral.",
}


# ==============================================================================
# 2. CARGA Y CACHE DE ARTEFACTOS
# ==============================================================================


@st.cache_resource(show_spinner=False)
def _load_cached_model(model_path: str) -> Any:
    """Carga el modelo una sola vez por sesión de Streamlit."""
    return load_model(model_path)


@st.cache_data(show_spinner=False)
def _dataset_metadata(data_path: str) -> dict[str, Any]:
    """Lee una sola vez los metadatos del dataset para construir el formulario."""
    return load_dataset_metadata(data_path)


def _run_training() -> None:
    """Ejecuta `scripts/train.py` para regenerar modelo y reportes."""
    with st.spinner("Entrenando el modelo. El proceso puede tardar entre 1 y 3 minutos..."):
        try:
            subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT_PATH)],
                cwd=str(TRAIN_SCRIPT_PATH.parent.parent),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() if exc.stderr else str(exc)
            st.error(f"El entrenamiento falló: {detail}")
            return
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario
            st.error(f"No se pudo ejecutar el entrenamiento: {exc}")
            return

    st.cache_resource.clear()
    st.session_state["training_message"] = (
        "Modelo entrenado y artefactos regenerados correctamente."
    )
    st.rerun()


def _load_model_or_show_message() -> Any | None:
    """
    Devuelve el modelo cargado o None si no está disponible.

    En lugar de romper la aplicación, explica cómo resolver cada situación:
    artefacto ausente (se puede entrenar desde la propia interfaz) o corrupto.
    """
    training_message = st.session_state.pop("training_message", None)
    if training_message:
        st.success(training_message)

    try:
        return _load_cached_model(str(DEFAULT_MODEL_PATH))
    except ModelArtifactNotFound as exc:
        st.warning(str(exc))
        if st.button("Entrenar el modelo ahora", type="primary"):
            _run_training()
        return None
    except ModelArtifactError as exc:
        st.error(str(exc))
        return None


# ==============================================================================
# 3. FORMULARIO
# ==============================================================================


def _render_numeric_field(
    container: Any,
    column: str,
    ranges: dict[str, tuple[float, float, float]],
    payload: dict[str, Any],
) -> None:
    """Renderiza un campo numérico con el rango observado en el dataset."""
    min_value, max_value, median = ranges[column]
    label = VARIABLE_LABELS.get(column, column)
    help_text = VARIABLE_HELP.get(column)

    if column in INTEGER_FEATURES:
        value = container.number_input(
            label,
            min_value=int(round(min_value)),
            max_value=int(round(max_value)),
            value=int(round(median)),
            step=1,
            key=f"numeric_{column}",
            help=help_text,
        )
        payload[column] = int(value)
    else:
        value = container.number_input(
            label,
            min_value=float(min_value),
            max_value=float(max_value),
            value=float(median),
            step=0.01,
            key=f"numeric_{column}",
            help=help_text,
        )
        payload[column] = float(value)


def _render_categorical_field(
    container: Any,
    column: str,
    options: dict[str, list[str]],
    defaults: dict[str, str],
    payload: dict[str, Any],
) -> None:
    """Renderiza un selector categórico con los valores presentes en el dataset."""
    values = options[column]
    default = defaults.get(column, values[0])
    index = values.index(default) if default in values else 0

    selected = container.selectbox(
        VARIABLE_LABELS.get(column, column),
        values,
        index=index,
        key=f"categorical_{column}",
        help=VARIABLE_HELP.get(column),
    )
    payload[column] = selected


def _render_client_form() -> dict[str, Any]:
    """Dibuja el formulario de captura y devuelve el registro del cliente."""
    metadata = _dataset_metadata(str(DEFAULT_DATA_PATH))
    options = metadata["categorical_options"]
    defaults = metadata["categorical_defaults"]
    ranges = metadata["numeric_ranges"]

    payload: dict[str, Any] = {}

    with st.form("client_profile_form"):
        left, right = st.columns(2)

        with left:
            st.markdown("**Variables de contacto**")
            for column in CLIENT_NUMERIC_FEATURES:
                if column in ranges:
                    _render_numeric_field(left, column, ranges, payload)

            st.markdown("**Contexto económico**")
            for column in ECONOMIC_FEATURES:
                if column in ranges:
                    _render_numeric_field(left, column, ranges, payload)

        with right:
            st.markdown("**Perfil del cliente**")
            for column in CLIENT_CATEGORICAL_FEATURES:
                if column in options:
                    _render_categorical_field(right, column, options, defaults, payload)

        submitted = st.form_submit_button("Predecir suscripción", type="primary")

    return payload if submitted else {}


# ==============================================================================
# 4. PESTAÑAS
# ==============================================================================


def _render_prediction_tab() -> None:
    """Pestaña principal: captura del cliente y resultado de la predicción."""
    st.subheader("Perfil del cliente")
    st.caption(
        "Los valores por defecto corresponden a la mediana (numéricas) y al valor más "
        "frecuente (categóricas) observados en el dataset de entrenamiento."
    )

    model = _load_model_or_show_message()
    if model is None:
        return

    payload = _render_client_form()
    if not payload:
        return

    try:
        result = predict_subscription(model, payload)
    except (KeyError, ModelArtifactError) as exc:
        st.error(f"No se pudo generar la predicción: {exc}")
        return

    st.divider()
    st.subheader("Resultado")

    left, right = st.columns([1, 2])

    with left:
        st.metric("Probabilidad de suscripción", f"{result['probability']:.1%}")
        st.progress(min(max(result["probability"], 0.0), 1.0))

    with right:
        if result["prediction"] == 1:
            st.success("El modelo predice que el cliente **sí se suscribe** al depósito.")
        else:
            st.warning("El modelo predice que el cliente **no se suscribe** al depósito.")

        if not uses_economic_features(model):
            st.caption(
                "Este modelo fue entrenado en el experimento *sin variables económicas*, "
                "por lo que los indicadores macroeconómicos no influyen en la predicción."
            )

        st.caption(
            "El modelo usa `class_weight='balanced'` para compensar el desbalance de clases "
            "(aproximadamente 89% de clientes que no suscriben). Por eso la probabilidad "
            "mostrada no es directamente comparable con la tasa base del 11%."
        )

    with st.expander("Ver los valores enviados al modelo"):
        st.json(payload)


def _render_performance_tab() -> None:
    """Pestaña de desempeño: métricas del entrenamiento y gráficos de evaluación."""
    st.subheader("Desempeño del modelo")

    metrics = load_metrics()
    if not metrics:
        st.info(
            "No hay métricas disponibles. Genera los artefactos con "
            "`uv run python scripts/train.py` y recarga la aplicación."
        )
        return

    summary = summarize_metrics(metrics)
    if summary.empty:
        st.info("El archivo de métricas existe pero no contiene resultados legibles.")
        return

    st.dataframe(summary, width="stretch")

    st.caption(
        "El modelo con mayor `f1_macro` es el que `scripts/train.py` guarda como "
        "`models/best_model.joblib`, que es el que consume esta aplicación."
    )

    st.markdown("#### Matriz de confusión del mejor modelo")
    confusion_path = DEFAULT_REPORTS_DIR / "confusion_matrix.png"
    if confusion_path.exists():
        st.image(str(confusion_path), width="stretch")
    else:
        st.info("Falta el archivo `reports/confusion_matrix.png`. Ejecuta el entrenamiento para generarlo.")

    st.markdown("#### Curvas ROC comparativas")
    roc_path = DEFAULT_REPORTS_DIR / "roc_curves.png"
    if roc_path.exists():
        st.image(str(roc_path), width="stretch")
    else:
        st.info("Falta el archivo `reports/roc_curves.png`. Ejecuta el entrenamiento para generarlo.")


def _render_help_tab() -> None:
    """Pestaña de ficha: qué hace el modelo, qué variables usa y cuáles son sus límites."""
    st.subheader("Ficha del modelo")

    st.markdown(
        """
El modelo estima la probabilidad de que un cliente suscriba un **depósito a plazo**
tras una llamada de marketing bancario. La variable objetivo es binaria
(`1 = suscribe`, `0 = no suscribe`) y el conjunto de datos es el *Bank Marketing*
(UCI / Moro et al., 2014), con 41.188 registros.
"""
    )

    st.markdown("#### Variables del cliente")
    st.dataframe(
        [
            {"variable": column, "significado": CLIENT_INFO.get(column, "")}
            for column in CLIENT_CATEGORICAL_FEATURES + CLIENT_NUMERIC_FEATURES
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Variables económicas (contexto del Banco de Portugal)")
    st.dataframe(
        [
            {"variable": column, "significado": ECONOMIC_INFO.get(column, "")}
            for column in ECONOMIC_FEATURES
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Supuestos y limitaciones")
    st.markdown(
        """
- **Sin fuga de información:** la variable `duration` (segundos de la llamada) se excluye
  estrictamente, porque solo se conoce al finalizar la llamada y no puede usarse para
  decidir a quién contactar.
- **Desbalance de clases:** cerca del 89% de los clientes no suscriben. Por eso el criterio
  de selección del mejor modelo es el `f1_macro` y no el accuracy.
- **Balanceo artificial:** los modelos usan `class_weight='balanced'`. El umbral de decisión
  por defecto (0.5) está calibrado sobre probabilidades reponderadas, por lo que no equivale
  a la probabilidad real de conversión.
- **Datos históricos:** el dataset corresponde a campañas de un banco europeo en un
  contexto económico concreto. Los indicadores macroeconómicos se mantienen constantes
  dentro de cada periodo, por lo que su aporte es de contexto y no de granularidad fina.
- **Sin datos en tiempo real:** la aplicación no consulta sistemas externos; opera sobre
  el artefacto entrenado y reproducible del repositorio.
"""
    )

    st.markdown("#### Reproducir los resultados")
    st.code(
        "uv sync\n"
        "uv run python scripts/train.py\n"
        "uv run streamlit run src/uasd_ds_2_final_project/app.py",
        language="bash",
    )


# ==============================================================================
# 5. PUNTO DE ENTRADA
# ==============================================================================


def main() -> None:
    """Construye la página de la aplicación."""
    st.set_page_config(page_title="Predicción de suscripciones", layout="wide")

    st.title("Predicción de suscripciones a depósitos a plazo")
    st.caption(
        "Proyecto final de Ciencia de Datos II (UASD) · Dataset Bank Marketing · "
        "Modelo entrenado por `scripts/train.py`"
    )

    prediction_tab, performance_tab, help_tab = st.tabs(
        ["Predicción", "Desempeño del modelo", "Ficha del modelo"]
    )

    with prediction_tab:
        _render_prediction_tab()

    with performance_tab:
        _render_performance_tab()

    with help_tab:
        _render_help_tab()


if __name__ == "__main__":
    main()
