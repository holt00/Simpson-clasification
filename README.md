# 🟡 Clasificación de personajes de Los Simpson con redes neuronales

Comparativa de **4 arquitecturas de deep learning** (red densa, CNN propia, Transfer Learning con Xception y Fine-Tuning con DenseNet201) y un **ensemble ponderado** para clasificar imágenes en **25 personajes** de Los Simpson. El mejor resultado alcanza un **98,3 % de accuracy en test**.

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20P100-20BEFF?logo=kaggle&logoColor=white)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

> Práctica 3 de **Minería de Datos** (curso 2024-2025): aprendizaje de modelos basados en redes neuronales con `tensorflow`.

<!-- Añade aquí una captura: la gráfica de comparación de accuracies o la cuadrícula de imágenes de ejemplo -->
<!-- ![Comparación de modelos](images/comparacion_modelos.png) -->

## 📊 Resultados

Accuracy sobre el conjunto de **test** (3.936 imágenes que ningún modelo vio durante el entrenamiento):

| Modelo | Descripción | Test accuracy |
|---|---|---|
| Red neuronal simple | Capas densas + BatchNorm + Dropout | 40,5 % |
| CNN propia | 5 bloques convolucionales (incl. convoluciones separables) | 94,3 % |
| Fine-Tuning | DenseNet201 (ImageNet), entrenamiento en 2 fases | 94,0 % |
| Transfer Learning | Xception (ImageNet) con todas las capas descongeladas | 98,2 % |
| **Ensemble** | Media ponderada de las probabilidades (TL 0,4 · CNN 0,3 · FT 0,2) | **98,3 %** |

**Qué se observa:**
- Las convoluciones lo cambian todo: de ~40 % (densa) a ~94 % (CNN) con una arquitectura pequeña (1,5 M de parámetros).
- El Transfer Learning con Xception es el mejor modelo individual, y `EarlyStopping` lo detuvo en la época 37 de 100.
- El ensemble mejora ligeramente al mejor modelo individual.
- DenseNet201 muestra algo de sobreajuste (≈99 % en train vs. ≈94 % en validación).

## 🗂️ Datos

Dataset de la competición de Kaggle `los-simpson`: imágenes de **25 personajes**, con clases desbalanceadas (domina `homer_simpson`).

| Conjunto | Imágenes | Cómo se obtiene |
|---|---|---|
| Train | 12.899 | 80 % del train original, estratificado |
| Validación | 3.238 | 20 % del train original, estratificado |
| Test | 3.936 | Conjunto de test de la competición (se descarta 1 imagen corrupta) |

## 🔧 Metodología

1. **Preparación de datos**: división estratificada con `split-folders` (semilla fija para reproducibilidad) y detección de una imagen corrupta en test.
2. **Análisis exploratorio**: comprobación de la estratificación, visualización de muestras y distribución de clases.
3. **Desbalanceo**: pesos por clase con `compute_class_weight('balanced')` aplicados en todos los entrenamientos.
4. **Data augmentation**: flip, rotación, zoom, contraste y brillo aleatorios, integrados como capas del modelo.
5. **Regularización**: Dropout, BatchNormalization, L2 y `GlobalAveragePooling2D`.
6. **Entrenamiento**: `ModelCheckpoint` sobre `val_loss`, `EarlyStopping` en los modelos pesados y *learning rate* con decaimiento exponencial en CNN y Xception.
7. **Fine-Tuning en dos fases** (DenseNet201): primero se entrenan las últimas 10 capas (lr = 1e-4); después se descongelan unas 500 capas más las 2 primeras con un lr mucho menor (1e-5).
8. **Ensemble**: media ponderada de las probabilidades de los modelos, sin reentrenar.

## 🚀 Cómo ejecutarlo

La libreta está pensada para **Kaggle con GPU (P100)**.

1. Sube `miner-a-pr-ctica-3.ipynb` a un notebook de Kaggle y añade como entrada el dataset de la competición `los-simpson`.
2. Activa el acelerador **GPU**.
3. Ejecuta todas las celdas (instala `plot_keras_history` y `split-folders` en la propia libreta).

> ⏱️ El tiempo total de ejecución fue de **5 h 45 min**, la mayor parte en los modelos de Transfer Learning y Fine-Tuning.

**Librerías:** `tensorflow` / `keras`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `split-folders`, `plot_keras_history`.

## 💡 Conclusiones

- Las CNN superan con claridad a las redes densas en clasificación de imágenes.
- Transfer Learning y Fine-Tuning son muy potentes, pero cuestan tiempo y recursos de GPU.
- Diseñar y ajustar una red sigue siendo un proceso iterativo de prueba y error; las horas de GPU limitadas de Kaggle impidieron explorar más configuraciones.
- Combinar modelos en un ensemble aporta robustez y una mejora pequeña pero consistente.

## 🔭 Posibles mejoras

- Búsqueda sistemática de hiperparámetros (p. ej. con KerasTuner).
- Matriz de confusión y análisis por clase para ver qué personajes se confunden.
- Calcular los pesos del ensemble a partir de la validación en lugar de fijarlos a mano.
- Probar arquitecturas más recientes (EfficientNet, ConvNeXt).

## 👥 Autores

- David López Lorenzo
- Carlos Martínez Jaén

Supervisado por el profesorado de la asignatura: Juan Carlos Alfaro Jiménez y José Antonio Gámez Martín.
