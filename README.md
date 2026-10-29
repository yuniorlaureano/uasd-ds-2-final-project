# Predicción de Respuesta a Campañas de Marketing Bancario

**Maestría en Inteligencia Artificial — Ciencia de Datos II (UASD)**  
**Modalidad:** Proyecto Aplicado de Ciencia de Datos

---

## 1. Dataset
- **Fuente:** [Bank Marketing Dataset (UCI / Moro et al., 2014)](https://archive.ics.uci.edu/dataset/222/bank+marketing), enriquecido con indicadores socioeconómicos del Banco de Portugal.
- **Ubicación en el repositorio:** `data/raw/bank-additional-full.csv` (41,188 registros, 20 atributos predictivos).

## 2. Objetivo del Proyecto
Desarrollar un modelo de Machine Learning capaz de **predecir con anticipación si un cliente contratará un depósito a plazo fijo** (`y = 1`) al ser contactado por telemarketing, permitiendo al banco priorizar clientes con alta probabilidad de conversión y reducir costos de contacto.

- **Variable objetivo (`y`):** `1: yes` (contrata), `0: no` (no contrata).
- **Control de fuga de datos (Data Leakage):** Se excluye estrictamente la variable `duration`, ya que es imposible conocer la duración de una llamada antes de realizarla.

## 3. Preguntas Clave del Proyecto
A partir del objetivo, el equipo busca responder:
1. **¿Es viable predecir la conversión del cliente antes del contacto?** ¿Qué efectividad (F1-score, ROC-AUC) alcanzamos superando una línea base ingenua?
2. **¿Qué factores pesan más en la decisión?** ¿Cuáles características demográficas o del historial bancario tienen mayor relación con la contratación?
3. **Hipótesis del contexto económico:** ¿En qué medida la situación económica del país (tasas Euribor, inflación, empleo) mejora la predicción frente a usar únicamente los datos del cliente?
4. **¿Qué algoritmo responde mejor?** ¿Un modelo lineal interpretable (Regresión Logística) o un ensamble no lineal (Random Forest)?

## 4. Modelos Implementados
Todos los modelos están integrados en `Pipeline` de scikit-learn con preprocesamiento unificado (`StandardScaler` + `OneHotEncoder`):
1. **Línea Base (Baseline):** `DummyClassifier` (estrategia más frecuente para validar la ganancia sobre el desbalance de clases).
2. **Modelo Lineal:** `LogisticRegression` (modelo interpretable con regularización y balanceo de clases).
3. **Modelo de Ensamble:** `RandomForestClassifier` (modelo no lineal para capturar interacciones complejas).

## 5. Ejecución del Pipeline y Pruebas
- **Ejecutar entrenamiento y generar artefactos (`reports/` y `models/`):**
  ```bash
  uv run python scripts/train.py
  ```
- **Ejecutar pruebas unitarias:**
  ```bash
  uv run pytest
  ```

## 6. Aplicación Web (Streamlit)
Interfaz que permite estimar en vivo la probabilidad de suscripción de un cliente potencial:
- **Generar el modelo y levantar la aplicación:**
  ```bash
  uv run python scripts/train.py
  uv run streamlit run src/uasd_ds_2_final_project/app.py
  ```
- **Documentación de la aplicación:** [`docs/streamlit_app.md`](docs/streamlit_app.md)
