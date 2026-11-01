# Model Card: Predicción de Respuesta a Campañas de Marketing Bancario

Documentación técnica y transparente del modelo de Machine Learning desarrollado para la asignatura **Ciencia de Datos II** de la **Maestría en Inteligencia Artificial (UASD)**.

---

## 1. Detalles del Modelo (Model Details)

* **Nombre del Proyecto:** Bank Marketing Response Predictor
* **Versión:** 1.0.0
* **Fecha de Creación:** Octubre 2026
* **Autores:** Luis Eduardo Puello Ureña & Equipo de Proyecto
* **Organización / Institución:** Universidad Autónoma de Santo Domingo (UASD) — Maestría en IA
* **Tipo de Modelo:** Clasificación Binaria Tabular
* **Algoritmos Implementados:** 
  * `DummyClassifier` (Baseline por clase más frecuente).
  * `LogisticRegression` (`class_weight='balanced'`, `max_iter=1000`).
  * `RandomForestClassifier` (`n_estimators=100`, `max_depth=10`, `class_weight='balanced'`).
* **Framework y Stack:** Python 3.10+, `scikit-learn`, `pandas`, `numpy`, `pytest`, `streamlit`.
* **Entorno de ejecución / Gestor de paquetes:** `uv` (`uv run python scripts/train.py`)

---

## 2. Uso Previsto (Intended Use)

### Casos de Uso Primarios
* **Optimización de Telemarketing:** Predecir con anticipación si un cliente potencial contratará un depósito a plazo fijo (`y = 1`) al ser contactado por vía telefónica.
* **Priorización de Cartera:** Permitir que la entidad financiera ordene su base de datos por probabilidad de conversión, reduciendo los costos operativos de contacto directo.

### Usos Fuera de Alcance (Out-of-Scope / Misuse)
* **Aprobación de Créditos o Riesgo Crediticio:** Este modelo **no** evalúa la capacidad de pago ni el scoring crediticio de un cliente.
* **Toma de decisiones automatizada sin supervisión:** No debe utilizarse como criterio único y excluyente para negar atención telefónica a segmentos de clientes sin revisión humana.

---

## 3. Datos de Entrenamiento y Evaluación (Data Overview)

* **Fuente:** *Bank Marketing Dataset* (UCI Machine Learning Repository / Moro et al., 2014), enriquecido con indicadores socioeconómicos del Banco de Portugal.
* **Ubicación:** `data/raw/bank-additional-full.csv` (41,188 registros, 20 atributos predictivos) y `data/processed/bank-additional-clean.csv`.
* **Variable Objetivo ($y$):** 
  * `1` (`yes`): Contrata depósito a plazo fijo (~11.3% de la muestra).
  * `0` (`no`): No contrata (~88.7% de la muestra).

### Preprocesamiento y Control de Fuga de Datos (Data Leakage)
1. **Fuga de Datos (Data Leakage):** Se excluye estrictamente la variable `duration` del pipeline de predicción, ya que la duración de la llamada solo se conoce *después* de realizar el contacto.
2. **Encapsulamiento en Pipelines:** Preprocesamiento unificado vía `ColumnTransformer`:
   * **Variables numéricas:** Escalado estándar (`StandardScaler`).
   * **Variables categóricas:** Codificación `OneHotEncoder` (conservando categorías `unknown`).
3. **Limpieza:** Recorte de espacios en blanco en cadenas de texto y auditoría reproducible mediante `scripts/audit_data.py`.

---

## 4. Métricas y Rendimiento Cuantitativo (Quantitative Analysis)

El dataset presenta un **desbalance de clases severo (~89% no vs. ~11% yes)**. Por ello, la métrica principal de evaluación es el **F1-Score macro** y el área bajo la curva **ROC-AUC**, en lugar del *Accuracy* directo.

| Modelo | Accuracy | Precision (Clase 1) | Recall (Clase 1) | F1-Score (Macro) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier** (Baseline) | ~0.89 | 0.00 | 0.00 | ~0.47 | 0.50 |
| **LogisticRegression** (`balanced`) | *[Inserta %]* | *[Inserta %]* | *[Inserta %]* | *[Inserta]* | *[Inserta]* |
| **RandomForestClassifier** (`balanced`) | *[Inserta %]* | *[Inserta %]* | *[Inserta %]* | *[Inserta]* | *[Inserta]* |

> **Nota:** Ejecuta `uv run python scripts/train.py` para extraer los números definitivos guardados en el reporte `reports/` y actualizar la tabla anterior.

---

## 5. Consideraciones Éticas, Sesgos y Limitaciones

* **Desbalance Socioeconómico:** El dataset contiene indicadores macroeconómicos (tasas Euribor, variación de empleo, índice de precios) pertenecientes a un contexto histórico y geográfico específico (Portugal). Cambios estructurales en la economía local pueden afectar el rendimiento del modelo (*Concept Drift*).
* **Sensibilidad a Falsos Negativos:** En una campaña bancaria, clasificar a un cliente potencial comprador como `0` (Falso Negativo) implica una pérdida de oportunidad comercial, mientras que un Falso Positivo solo genera costo de llamada. El ajuste del umbral de decisión (*Decision Threshold*) debe responder al costo marginal por llamada del banco.

---

## 6. Instrucciones de Uso y Ejemplo en Python

### 1. Requisitos e Instalación
```bash
# Clonar repositorio e instalar dependencias con uv
uv sync

## 7.Entrenamiento y Pruebas Bash
## Generar artefatos en reports/ y models/
uv run python scripts/train.py

## 8. Ejecutar la suite de pruebas unitarias (16+ test cases)
uv run pytest

## Inferencia con la App de Streamlit
uv run streamlit run src/uasd_ds_2_final_project/app.py

## Snippet de Código para Cargar y Predecir
Python

import joblib
import pandas as pd

# 1. Cargar el pipeline entrenado (preprocesador + modelo)
pipeline = joblib.load("models/random_forest_pipeline.joblib")

# 2. Construir muestra de prueba (sin incluir la variable 'duration')
sample_client = pd.DataFrame([{
    'age': 35,
    'job': 'technician',
    'marital': 'single',
    'education': 'university.degree',
    'default': 'no',
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'month': 'may',
    'day_of_week': 'mon',
    'campaign': 1,
    'pdays': 999,
    'previous': 0,
    'poutcome': 'nonexistent',
    'emp.var.rate': -1.8,
    'cons.price.idx': 92.893,
    'cons.conf.idx': -46.2,
    'euribor3m': 1.299,
    'nr.employed': 5099.1
}])

# 3. Predicción de probabilidad y clase
prob = pipeline.predict_proba(sample_client)[0][1]
prediction = pipeline.predict(sample_client)[0]

print(f"Probabilidad de suscripción: {prob:.2%}")
print(f"Predicción (1=Suscrito, 0=No suscrito): {prediction}")

---

### ¿Qué paso sigue?
Solo debes abrir los JSONs o logs que genera tu comando `uv run python scripts/train.py` (en la carpeta `reports/`) y rellenar la tabla del punto **4 (Métricas y Rendimiento)** con los valores finales.

