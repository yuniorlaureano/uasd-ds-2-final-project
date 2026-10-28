"""
Módulo de Datos y Preprocesamiento
==================================
Este módulo contiene todas las funciones necesarias para cargar, validar,
limpiar y transformar el dataset de Marketing Bancario (Bank Marketing).

Estructura diseñada siguiendo las mejores prácticas de Cookiecutter Data Science.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==============================================================================
# 1. DEFINICIÓN DE GRUPOS DE VARIABLES
# ==============================================================================

# Variables numéricas propias del cliente y de la gestión de contacto
CLIENT_NUMERIC_FEATURES = [
    "age",       # Edad del cliente
    "campaign",  # Número de contactos realizados durante esta campaña
    "pdays",     # Días transcurridos desde el último contacto de una campaña previa (999 = no contactado)
    "previous",  # Número de contactos realizados antes de esta campaña
]

# Variables categóricas del perfil del cliente y del contexto de la llamada
CLIENT_CATEGORICAL_FEATURES = [
    "job",          # Ocupación ('admin.', 'blue-collar', 'technician', etc.)
    "marital",      # Estado civil ('married', 'single', 'divorced')
    "education",    # Nivel educativo ('university.degree', 'high.school', etc.)
    "default",      # ¿Tiene créditos en mora? ('no', 'yes', 'unknown')
    "housing",      # ¿Tiene préstamo hipotecario? ('no', 'yes', 'unknown')
    "loan",         # ¿Tiene préstamo personal? ('no', 'yes', 'unknown')
    "contact",      # Tipo de comunicación del contacto ('cellular', 'telephone')
    "month",        # Último mes de contacto ('may', 'jul', 'aug', etc.)
    "day_of_week",  # Último día de contacto ('mon', 'tue', 'wed', etc.)
    "poutcome",     # Resultado de la campaña de marketing previa ('failure', 'nonexistent', 'success')
]

# Variables socioeconómicas a nivel país (proporcionadas por el Banco de Portugal)
# Estas variables nos permiten responder la pregunta de investigación:
# ¿Mejora la predicción al incluir el contexto económico frente a solo datos del cliente?
ECONOMIC_FEATURES = [
    "emp.var.rate",    # Tasa de variación del empleo (indicador trimestral)
    "cons.price.idx",   # Índice de precios al consumidor (mensual)
    "cons.conf.idx",    # Índice de confianza del consumidor (mensual)
    "euribor3m",        # Tasa Euribor a 3 meses (diaria)
    "nr.employed",      # Número de empleados en miles (trimestral)
]

# Variable objetivo a predecir: ¿el cliente se suscribió al depósito? ('yes' / 'no')
TARGET_COLUMN = "y"

# PRECAUCIÓN DE FUGA DE INFORMACIÓN (Data Leakage):
# La variable 'duration' (duración de la llamada en segundos) solo se conoce
# DESPUÉS de que la llamada finaliza. Si la usamos para predecir si llamar o no
# a un cliente, tendríamos un modelo irrealista que no funcionaría en producción.
# Por tanto, se excluye de forma estricta.
LEAKAGE_COLUMNS = ["duration"]


# ==============================================================================
# 2. CARGA Y VALIDACIÓN DE DATOS
# ==============================================================================

def load_raw_data(filepath: Path | str) -> pd.DataFrame:
    """
    Carga el archivo CSV crudo desde la ruta especificada.
    
   El dataset de Bank Marketing utiliza punto y coma (';')
    como separador en lugar de la coma tradicional (',').
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {path}")
    
    # sep=';' es fundamental para este dataset europeo
    df = pd.read_csv(path, sep=";")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Asegura la calidad y reproducibilidad del dataset verificando que no esté vacío
    y contenga todas las columnas obligatorias requeridas por el proyecto.
    """
    if df.empty:
        raise ValueError("El DataFrame cargado está vacío. Verifica la ruta del archivo.")
    
    # 1. Validar que la variable objetivo esté presente
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Falta la columna objetivo obligatoria: '{TARGET_COLUMN}'")
    
    # 2. Validar que todas las variables predictoras esperadas existan
    expected_columns = set(CLIENT_NUMERIC_FEATURES + CLIENT_CATEGORICAL_FEATURES + ECONOMIC_FEATURES)
    missing_columns = expected_columns - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias en el dataset: {missing_columns}")


# ==============================================================================
# 3. PREPARACIÓN Y DIVISIÓN ESTRATIFICADA
# ==============================================================================

def prepare_features_and_target(
    df: pd.DataFrame,
    include_economic_features: bool = True,
    drop_leakage: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Separa las variables predictoras (X) de la variable objetivo (y).
    
    Parámetros didácticos:
    ----------------------
    include_economic_features: bool
        - True: Incluye variables macroeconómicas (Modelo completo).
        - False: Solo incluye datos del cliente (Modelo base para responder
                 a la pregunta de investigación sobre el impacto de la economía).
    drop_leakage: bool
        - True: Elimina 'duration' para evitar fuga de información temporal.
    
    Retorna:
    --------
    X: DataFrame con las variables predictoras limpias.
    y: Serie binaria (1 para 'yes', 0 para 'no').
    numeric_features: Lista con los nombres de variables numéricas activas.
    categorical_features: Lista con los nombres de variables categóricas activas.
    """
    validate_data(df)

    numeric_features = list(CLIENT_NUMERIC_FEATURES)
    categorical_features = list(CLIENT_CATEGORICAL_FEATURES)

    # Agregamos variables macroeconómicas según el experimento
    if include_economic_features:
        numeric_features.extend(ECONOMIC_FEATURES)

    # Lista total de columnas que seleccionaremos
    selected_columns = numeric_features + categorical_features

    # Eliminamos cualquier variable con fuga de datos (como 'duration')
    if drop_leakage:
        selected_columns = [col for col in selected_columns if col not in LEAKAGE_COLUMNS]

    X = df[selected_columns].copy()
    
    # Convertimos la variable objetivo a binaria: 'yes' -> 1, 'no' -> 0
    y = (df[TARGET_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(int)

    return X, y, numeric_features, categorical_features


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de Entrenamiento (Train) y Prueba (Test).
    
    Usamos 'stratify=y' porque en marketing bancario las clases están desbalanceadas
    (aprox. 89% dijeron 'no' y solo 11% dijeron 'yes').
    La estratificación garantiza que tanto en Train como en Test se mantenga
    exactamente esa misma proporción de 89% vs 11%, evitando sesgos en la evaluación.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ==============================================================================
# 4. PREPROCESADOR DE SCIKIT-LEARN (ColumnTransformer)
# ==============================================================================

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Construye el transformador de columnas que se integrará dentro del Pipeline final.
    
    ¿Qué hace y por qué?
    1. StandardScaler para numéricas:
       Resta la media y divide entre la desviación estándar. Esto evita que una variable
       con escala grande (ej. 'pdays' con valor 999) opaque a variables con escalas
       pequeñas (ej. 'previous' con valores 0, 1, 2).
       
    2. OneHotEncoder para categóricas:
       Convierte texto como 'admin.', 'blue-collar' en columnas de ceros y unos (0/1).
       'handle_unknown="ignore"' previene errores en producción si aparece una
       categoría nueva que no existía en el entrenamiento.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric_pipeline", numeric_transformer, numeric_features),
            ("categorical_pipeline", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
