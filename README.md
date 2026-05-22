# Clasificacion de Enfermedades en Hojas de Cultivos
### Proyecto — Fundamentos de Deep Learning | UdeA 2026-1

Clasificacion automatica de enfermedades en plantas a partir de fotos de hojas, usando CNNs y Transfer Learning sobre el dataset PlantVillage (54,305 imagenes, 38 clases).

---

## Video de presentacion

> **[Enlace al video en YouTube — por agregar]**

---

## Estructura del repositorio

| Archivo | Descripcion |
|---|---|
| `01 - Exploracion de datos.ipynb` | Analisis del dataset, distribucion de clases y visualizacion de muestras |
| `02 - Preprocesado.ipynb` | Pipeline de augmentation, normalizacion y pesos de clase |
| `03 - Modelo baseline CNN.ipynb` | CNN desde cero como punto de comparacion |
| `04 - Transfer Learning VGG16.ipynb` | Feature extraction y fine-tuning con VGG16 |
| `05 - Transfer Learning ResNet50.ipynb` | Feature extraction y fine-tuning con ResNet50 |
| `06 - Transfer Learning InceptionV3.ipynb` | Feature extraction y fine-tuning con InceptionV3 |
| `07 - Comparacion y GradCAM.ipynb` | Comparativa de modelos, GradCAM y analisis de errores |
| `ENTREGA1.pdf` | Documento de la primera entrega |

---

## Dataset

PlantVillage — 54,305 imagenes JPEG, 256x256 px, 38 clases (14 cultivos con distintas enfermedades y estado saludable).

Se carga directamente desde `tensorflow_datasets`, no requiere descarga manual:

```python
import tensorflow_datasets as tfds
ds, info = tfds.load('plant_village', with_info=True, as_supervised=True)
```

Split usado: 70% entrenamiento / 15% validacion / 15% test.

---

## Como ejecutar los notebooks

Todos los notebooks estan pensados para correr en Google Colab con GPU T4.

1. Abrir el notebook en Google Colab
2. Ejecutar todas las celdas en orden
3. Los notebooks descargan automaticamente los archivos necesarios desde una carpeta publica de Drive usando `gdown`
4. Los modelos entrenados quedan en `/content/plantvillage/` dentro de la sesion de Colab

Los notebooks 04, 05 y 06 tienen una celda opcional antes del entrenamiento que permite cargar el modelo directamente si ya fue entrenado antes, sin tener que volver a correr las fases de entrenamiento.

---

## Resultados

| Modelo | Test Accuracy |
|---|---|
| CNN Baseline | 62.77% |
| VGG16 | 66.70% |
| ResNet50 | 97.75% |
| InceptionV3 | 71.76% |

---

## Referencias

- Mohanty, S.P., Hughes, D.P. & Salathe, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 7, 1419.
- Too, E.C. et al. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. Computers and Electronics in Agriculture, 161, 272-279.
- Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017.
