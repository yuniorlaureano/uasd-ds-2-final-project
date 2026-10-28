"""
Módulo de Modelos y Pipelines de Machine Learning
=================================================
Este módulo define las fábricas de modelos para el proyecto final.
Cada función construye un `Pipeline` completo de scikit-learn que une
el preprocesador (escalado + codificación) directamente con el algoritmo.

Modelos implementados:
1. DummyClassifier (Línea base / Baseline)
2. LogisticRegression (Modelo lineal interpretable)
3. RandomForestClassifier (Modelo de ensamble no lineal)
"""

from __future__ import annotations

from typing import Callable
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ==============================================================================
# 1. MODELO BASELINE (DummyClassifier)
# ==============================================================================

def build_dummy_pipeline(
    preprocessor: ColumnTransformer,
    strategy: str = "most_frequent",
) -> Pipeline:
    """
    Construye el pipeline de línea base (Baseline) usando DummyClassifier.

    Propósito didáctico:
    --------------------
    En datos desbalanceados (como este dataset bancario con ~89% de 'no'),
    un modelo ingenuo que siempre prediga 'no' obtendrá un 89% de Accuracy.
    El DummyClassifier sirve como punto de referencia mínimo: cualquier
    modelo real (regresión logística o random forest) debe superar
    significativamente al Dummy en métricas como F1-score macro y ROC-AUC.
    """
    classifier = DummyClassifier(strategy=strategy)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


# ==============================================================================
# 2. MODELO LINEAL (LogisticRegression)
# ==============================================================================

def build_logistic_regression_pipeline(
    preprocessor: ColumnTransformer,
    random_state: int = 42,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
) -> Pipeline:
    """
    Construye el pipeline para Regresión Logística.

    Propósito didáctico:
    --------------------
    La Regresión Logística es el estándar de la industria financiera para
    clasificación tabular binaria por su alta interpretabilidad (pesos/coeficientes).

    Parámetros clave:
    - max_iter=1000: Permite que el optimizador converja sin advertencias.
    - class_weight='balanced': Ajusta automáticamente los pesos de las clases
      de forma inversamente proporcional a sus frecuencias. Esto ayuda al modelo
      a prestarle mayor atención a los clientes que dijeron 'yes' (la clase minoritaria).
    - random_state=42: Asegura que los resultados sean exactamente reproducibles.
    """
    classifier = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


# ==============================================================================
# 3. MODELO DE ENSAMBLE (RandomForestClassifier)
# ==============================================================================

def build_random_forest_pipeline(
    preprocessor: ColumnTransformer,
    n_estimators: int = 100,
    max_depth: int | None = 10,
    random_state: int = 42,
    class_weight: str | None = "balanced",
) -> Pipeline:
    """
    Construye el pipeline para Random Forest (Bosque Aleatorio).

    Propósito didáctico:
    --------------------
    Random Forest combina múltiples árboles de decisión mediante Bagging.
    Es capaz de capturar relaciones no lineales y complejas entre los indicadores
    macroeconómicos y el perfil del cliente sin asumir que los datos siguen
    una distribución lineal.

    Parámetros clave:
    - n_estimators=100: Número de árboles en el bosque.
    - max_depth=10: Limita la profundidad de cada árbol para evitar sobreajuste (overfitting).
    - class_weight='balanced': Compensa el desbalanceo de clases.
    - n_jobs=-1: Utiliza todos los núcleos del procesador para entrenar en paralelo.
    """
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


# ==============================================================================
# 4. REGISTRO DE FÁBRICAS DE MODELOS
# ==============================================================================

def get_model_factories() -> dict[str, Callable[[ColumnTransformer], Pipeline]]:
    """
    Retorna un diccionario con las funciones constructoras de cada modelo.
    
    Permite iterar sobre todos los modelos de forma limpia en el script de
    entrenamiento, tal como en el laboratorio de la Unidad 2:
    `for name, factory in get_model_factories().items(): ...`
    """
    return {
        "dummy_baseline": build_dummy_pipeline,
        "logistic_regression": build_logistic_regression_pipeline,
        "random_forest": build_random_forest_pipeline,
    }
