# Ficha del dataset: Bank Marketing (bank-additional-full)

## Identificación y procedencia

| Campo | Descripción |
|---|---|
| Nombre | Bank Marketing (con contexto social y económico) |
| Archivo integrado | [`data/raw/bank-additional-full.csv`](../data/raw/bank-additional-full.csv) |
| Diccionario original | [`data/raw/bank-additional-names.txt`](../data/raw/bank-additional-names.txt) |
| Fuente | [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| Autores | Sérgio Moro, Paulo Cortez y Paulo Rita |
| Publicación | Moro, S., Cortez, P. y Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems. [doi:10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001) |
| Dominio | Respuesta de clientes a campañas de telemarketing para depósitos bancarios a plazo |
| Unidad de análisis | Registro de un cliente/contacto de la campaña |
| Periodo reportado | Mayo de 2008 a noviembre de 2010 |
| Dimensión fuente | 41,188 filas x 21 columnas: 20 variables predictoras y `y` |
| Formato de origen | CSV separado por punto y coma (`;`) |

El archivo completo contiene información del perfil del cliente, contactos e
historial de campañas, además de cinco indicadores socioeconómicos nacionales
publicados por el Banco de Portugal.

## Objetivo y uso analítico

- **Problema:** clasificación binaria de la respuesta a una campaña.
- **Objetivo (`y`):** si el cliente suscribió un depósito a plazo.
- **Codificación del archivo procesado:** `no = 0`, `yes = 1`.
- **Uso previsto:** análisis académico de propensión para apoyar la priorización
  de contactos de marketing, con revisión humana.
- **No es un uso previsto:** aprobar créditos, definir elegibilidad financiera o
  tomar decisiones automatizadas de alto impacto.

## Diccionario de variables

| Variable(s) | Tipo | Descripción |
|---|---|---|
| `age` | Numérica | Edad del cliente |
| `job` | Categórica | Ocupación |
| `marital` | Categórica | Estado civil; `divorced` agrupa divorcio y viudez |
| `education` | Categórica | Nivel educativo |
| `default` | Categórica | Indicador de crédito en mora |
| `housing` | Categórica | Indicador de préstamo hipotecario |
| `loan` | Categórica | Indicador de préstamo personal |
| `contact` | Categórica | Medio de comunicación del contacto |
| `month`, `day_of_week` | Categóricas | Mes y día de la semana del contacto |
| `duration` | Numérica | Duración de la última llamada, en segundos |
| `campaign` | Numérica | Contactos durante la campaña actual, incluido el último |
| `pdays` | Numérica | Días desde un contacto anterior; `999` indica que no hubo contacto previo |
| `previous` | Numérica | Contactos antes de la campaña actual |
| `poutcome` | Categórica | Resultado de la campaña de marketing anterior |
| `emp.var.rate` | Numérica | Tasa de variación del empleo, indicador trimestral |
| `cons.price.idx` | Numérica | Índice de precios al consumidor, indicador mensual |
| `cons.conf.idx` | Numérica | Índice de confianza del consumidor, indicador mensual |
| `euribor3m` | Numérica | Tasa Euribor a tres meses, indicador diario |
| `nr.employed` | Numérica | Número de personas empleadas, indicador trimestral |
| `y` | Categórica binaria | Suscripción del depósito a plazo: `yes`/`no` |

Las categorías `unknown` representan información no disponible en algunos
campos; no son celdas vacías y se conservan explícitamente. `pdays = 999` es un
código especial, no una duración literal.

## Auditoría de calidad

La auditoría es reproducible con `uv run python scripts/audit_data.py`. El
reporte detallado se guarda en
[`reports/data_audit.json`](../reports/data_audit.json).

| Revisión sobre la fuente | Resultado |
|---|---:|
| Filas y columnas | 41,188 x 21 |
| Valores faltantes (`NaN`) | 0 |
| Valores numéricos no finitos | 0 |
| Etiquetas observadas de `y` | `no`, `yes` |
| Registros `no` | 36,548 (88.7346 %) |
| Registros `yes` | 4,640 (11.2654 %) |
| Filas exactamente repetidas | 12 |

Hay valores `unknown` en variables categóricas: `job` (330), `marital` (80),
`education` (1,731), `default` (8,597), `housing` (990) y `loan` (990).

Las 12 filas exactamente repetidas se **conservan**: el archivo no incluye un
identificador que permita distinguir duplicación accidental de dos contactos
con atributos y resultado idénticos. Las observaciones potencialmente atípicas
por la regla de 1.5 x IQR se señalan en el JSON y no se eliminan automáticamente.
El desbalance del objetivo debe considerarse al evaluar, prestando atención a
recall y precisión de la clase positiva.

## Limpieza y procesamiento

El script genera
[`data/processed/bank-additional-clean.csv`](../data/processed/bank-additional-clean.csv):

1. Recorta espacios iniciales/finales de campos de texto.
2. Codifica la variable objetivo como `no = 0`, `yes = 1`.
3. Conserva todas las filas, los valores `unknown` y las columnas originales,
   incluida `duration`, para facilitar trazabilidad con la fuente.
4. No imputa ni elimina extremos porque la auditoría no demuestra que sean
   errores.

Aunque `duration` se conserva en este archivo procesado, se excluye de las
variables predictoras del proyecto: solo se conoce al terminar la llamada y su
uso causaría fuga de información para una decisión previa al contacto.

## Limitaciones

- Los datos representan campañas históricas de un banco portugués y no
  demuestran generalización a otros países, bancos, poblaciones o periodos.
- Las cinco variables económicas son indicadores nacionales con distintas
  granularidades; no son mediciones individuales de los clientes.
- `unknown` puede representar diferencias en los procesos de recolección.
- Una asociación predictiva no demuestra que contactar a una persona cause una
  suscripción ni garantiza que el contacto sea beneficioso.

## Referencias

- [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Moro, S., Cortez, P. y Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems. [doi:10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001)
- [README del proyecto](../README.md)
