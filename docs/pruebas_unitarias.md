# 16 pruebas unitarias adicionales

Los casos usan datos sintéticos, archivos temporales de pytest y modelos mínimos.
No necesitan entrenar modelos ni ejecutar el pipeline sobre el dataset completo.

| Archivo | Cantidad | Comportamientos verificados |
| --- | ---: | --- |
| `tests/test_data_edge_cases.py` | 7 | Archivo ausente; CSV con delimitadores entre comillas; predictores ausentes; normalización de etiquetas sin modificar la fuente; partición estratificada sin solapamiento y con índices alineados; categorías nuevas; escalado con estadísticas del entrenamiento. |
| `tests/test_audit_edge_cases.py` | 4 | Objetivo ausente; etiquetas nulas; conservación de duplicados y datos originales; auditoría de faltantes, infinitos, atípicos, categorías desconocidas, clases y huella SHA-256. |
| `tests/test_inference_edge_cases.py` | 5 | Artefacto serializado de tipo incorrecto; pipeline sin metadatos de entrada; métricas válidas en UTF-8; JSON truncado; JSON con raíz de tipo incorrecto. |
| **Total** | **16** | |

Ejecutar únicamente estas 16 pruebas:

```bash
uv run pytest -q tests/test_data_edge_cases.py tests/test_audit_edge_cases.py tests/test_inference_edge_cases.py
```

Ejecutar la suite completa, incluyendo las pruebas existentes:

```bash
uv run pytest -q
```

Las pruebas existentes de la interfaz que requieren un modelo entrenado se omiten
si no existe `models/best_model.joblib`, según la condición definida en `tests/test_app.py`.
