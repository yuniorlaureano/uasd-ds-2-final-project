# Ficha del dataset: Banknote Authentication

## Identificación y procedencia

| Campo | Descripción |
|---|---|
| Nombre | Banknote Authentication |
| Origen | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication) |
| Autor | Volker Lohweg, University of Applied Sciences, Ostwestfalen-Lippe |
| Licencia | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| Dominio | Clasificación de billetes auténticos y falsificados a partir de características extraídas de imágenes |
| Unidad de análisis | Una fila representa las características de una imagen de un billete auténtico o falsificado |
| Archivo fuente integrado | [`data/raw/data_banknote_authentication.txt`](../data/raw/data_banknote_authentication.txt) |
| Archivo procesado | [`data/processed/banknote_authentication_clean.csv`](../data/processed/banknote_authentication_clean.csv) |
| Tamaño del archivo fuente | 45,029 bytes (aprox. 44.0 KiB) |
| Dimensión fuente | 1,372 filas × 5 columnas |
| Dimensión procesada | 1,348 filas × 5 columnas, tras retirar 24 filas duplicadas exactas |
| Dimensión temporal | No contiene fecha, hora ni otra dimensión temporal |

## Descripción del dominio

Los datos se obtuvieron de imágenes de billetes auténticos y de billetes falsificados que imitaban a los auténticos. Para digitalizarlos se usó una cámara industrial empleada habitualmente en la inspección de impresiones. Las imágenes se prepararon con una resolución de 400 × 400 píxeles, en escala de grises y con una resolución aproximada de 660 ppp. A partir de esas imágenes se extrajeron características mediante una transformada *wavelet*.

El archivo de este proyecto **no contiene las imágenes**: contiene cuatro características numéricas derivadas de ellas y una etiqueta de clase.

## Diccionario de variables

| Variable | Rol | Tipo | Descripción | Unidad | Valores faltantes |
|---|---|---|---|---|---|
| `variance` | Característica | Continua (float) | Varianza de la imagen transformada mediante *wavelet* | No aplica | No |
| `skewness` | Característica | Continua (float) | Asimetría de la imagen transformada mediante *wavelet* | No aplica | No |
| `curtosis` | Característica | Continua (float) | Curtosis de la imagen transformada mediante *wavelet* | No aplica | No |
| `entropy` | Característica | Continua (float) | Entropía de la imagen | No aplica | No |
| `class` | Objetivo | Entero (int) | Clase del billete: `0` = auténtico; `1` = falsificado | No aplica | No |

Las variables predictoras son medidas numéricas derivadas de las imágenes; no son unidades físicas de los billetes. En los datos fuente, las cuatro características son `float64` y `class` es `int64`.

## Decisión analítica

- **Problema:** clasificar un billete como auténtico o falsificado.
- **Objetivo (`class`):** detectar billetes falsificados; la clase positiva es `1`.
- **Error más costoso:** un falso negativo —predecir `0` (auténtico) cuando el billete realmente pertenece a la clase `1` (falsificado)—, porque permite que un billete falso sea aceptado.
- **Usuario previsto:** personal o sistemas de inspección de entidades bancarias.
- **Origen de las características:** un software procesa las imágenes y genera las mediciones; el modelo recibe esas características para clasificar el billete.

## Auditoría y procesamiento

La auditoría se ejecutó sobre el archivo fuente integrado en el proyecto.

| Revisión | Resultado |
|---|---:|
| Filas y columnas iniciales | 1,372 × 5 |
| Valores faltantes | 0 |
| Valores no finitos en las características | 0 |
| Etiquetas observadas | `0` y `1` |
| Filas duplicadas exactas | 24 |
| Filas tras deduplicar | 1,348 |
| Clase `0` antes → después | 762 → 738 |
| Clase `1` antes → después | 610 → 610 |

Se eliminaron las 24 filas duplicadas exactas porque no aportan observaciones distintas y podrían sobrerrepresentar registros repetidos. No se imputaron datos, pues no hay valores faltantes, ni se transformaron o eliminaron valores extremos. Sobre los datos procesados, el criterio de 1.5 × IQR señaló posibles extremos en `curtosis` (59 registros) y `entropy` (32); se conservaron al no haber evidencia de que sean errores. La copia procesada mantiene las cinco columnas y se guarda como CSV.

La clase está algo desbalanceada: antes de deduplicar, 762 registros son auténticos (55.54 %) y 610 falsificados (44.46 %). Después de deduplicar, hay 738 auténticos (54.75 %) y 610 falsificados (45.25 %). La evaluación de un modelo debe prestar atención al *recall* de la clase `1` y al costo de los falsos negativos, además de la exactitud global.

El procedimiento reproducible está en [`banknote_authentication.ipynb`](../notebooks/banknote_authentication.ipynb).

## Glosario

1. **Transformada *wavelet* (u ondícula):** método matemático que representa una señal como combinaciones de ondas localizadas, útil para analizar características a distintas escalas.
2. **Píxel:** elemento mínimo de una imagen digital.
3. **Digitalización:** conversión de información analógica —por ejemplo, una imagen impresa— a una representación digital que puede almacenarse y procesarse.
4. **Escala de grises:** representación de una imagen con tonos que van del negro al blanco.
5. **ppp:** píxeles por pulgada; expresa la densidad de píxeles de una imagen.
6. **Entropía:** medida de la dispersión o complejidad de una distribución. En este dataset es una característica de imagen, no una medición directa de la entropía termodinámica.
