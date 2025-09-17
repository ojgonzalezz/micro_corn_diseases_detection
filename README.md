# Detección de enfermedades en cultivos de maíz

Este documento presenta los hallazgos de la primera fase del proyecto, centrada en el análisis y la comprensión de los datos iniciales.

---

## Problema y contexto

Las enfermedades del maíz, como la roya común, el tizón foliar y la mancha gris, representan una amenaza crítica para la seguridad alimentaria. El diagnóstico tradicional mediante inspección visual es un proceso lento, subjetivo y dependiente de la pericia del observador. Este proyecto busca validar la viabilidad de un sistema de diagnóstico automatizado mediante Inteligencia Artificial para superar estas limitaciones.

---

## Dataset inicial
Para el análisis, se utilizó el dataset público **"Corn or Maize Leaf Disease Dataset"** de Kaggle, una compilación de imágenes de las fuentes PlantVillage y PlantDoc.

* **Total de imágenes:** 4,188
* **Formato:** JPEG (.jpg)
* **Distribución de clases (Inicial):**
    * Roya Común (Common Rust): 1,306 imágenes (31.2%)
    * Sana (Healthy): 1,162 imágenes (27.7%)
    * Tizón (Blight): 1,146 imágenes (27.4%)
    * Mancha Gris (Gray Leaf Spot): 574 imágenes (13.7%)

> **Observación:** El dataset inicial presenta un notable desbalance, con la clase **"Mancha Gris"** significativamente subrepresentada. Este hallazgo es fundamental para las siguientes etapas del proyecto.

---

### Hallazgos del análisis exploratorio de datos (EDA)

### Validación e integridad de datos
Se realizó una validación estructural del dataset para confirmar la cantidad de clases, el número de imágenes y la integridad de los archivos. Se encontró y corrigió una inconsistencia de formato (un archivo **.jpeg** en lugar de **.jpg**) en la clase "Blight", asegurando la homogeneidad del conjunto de datos.

### Análisis cualitativo visual
La inspección de muestras aleatorias reveló una buena calidad de imagen general (nitidez y enfoque). Se destacó una alta variabilidad en iluminación, escala y ángulos de captura, lo cual es beneficioso para entrenar un modelo más robusto y generalizable.

> **Desafío:** Se observó una alta similitud morfológica entre las lesiones en etapas avanzadas de **"Mancha Gris"** y **"Tizón"**, lo que anticipa el principal reto de clasificación para el modelo de visión artificial.

### Análisis cuantitativo de características físicas
* **Dimensiones:** Se confirmó una considerable variabilidad en el tamaño (alto y ancho) de las imágenes, lo que fundamenta la necesidad de un paso de redimensionamiento estándar antes de alimentar el modelo.
* **Distribución de color:** El análisis de histogramas de color, particularmente en el canal verde, demostró ser un rasgo altamente discriminatorio. Las hojas sanas ("Healthy") mostraron un perfil de color verde único y vibrante, claramente distinto al de las hojas enfermas. Esto valida el potencial del color como una característica potente para la clasificación automática y justifica la necesidad de normalizar los valores de los píxeles.

---

## ⚙️ Metodología y Arquitectura del Modelo

El proyecto siguió un flujo de trabajo iterativo y completo de Machine Learning:

1.  **Análisis Exploratorio de Datos (EDA):** Se analizaron los datasets, revelando un **desbalance de clases** significativo y una alta similitud visual entre las lesiones de *Blight* y *Gray Leaf Spot*, anticipando un desafío de clasificación.

2.  **Preprocesamiento y Balanceo:** Se aplicó **submuestreo (undersampling)** para crear un dataset perfectamente balanceado de 4,580 imágenes (1,145 por clase). Posteriormente, se dividió de forma estratificada en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%). Se construyó un pipeline de datos para aplicar **aumento de datos en tiempo real** (rotaciones, zoom, etc.) al conjunto de entrenamiento.

3.  **Modelado y Entrenamiento (Iteración 1):**
    * Se implementó una arquitectura de **Transfer Learning** utilizando **VGG16** pre-entrenado en ImageNet como base.
    * Se realizaron 3 entrenamientos con distintos parámetros
      -  Entrenamiento 1, con los siguientes parámetros: IMAGE_SIZE = (224, 224),  BATCH_SIZE = 32,  NUM_CLASSES = 4 EPOCHS = 25, se obtuvo una precision  de **86.05%**.
      -  Entrenamiento 2, con los siguientes parámetros: IMAGE_SIZE = (224, 224),  BATCH_SIZE = 64,  NUM_CLASSES = 4 EPOCHS = 15, se obtuvo una precision  de **63.83%**.
      -  Entrenamiento 3, con los siguientes parámetros: IMAGE_SIZE = (224, 224),  BATCH_SIZE = 16,  NUM_CLASSES = 4 EPOCHS = 10, se obtuvo una precision  de **87.50%**.
        A continuación evidencias de las ejecuciones usando MlFow
<img width="1908" height="673" alt="image" src="https://github.com/user-attachments/assets/b5ad5cd2-9dff-47b2-911b-efad24cefc54" />

Aquí podemos ver el resultado de la validación y la matriz de confusion del modelo elegido el cual fue el obtenido en el Entrenamiento 2
<img width="787" height="467" alt="image" src="https://github.com/user-attachments/assets/8a6c5e98-9150-4c62-a513-56f5435be573" />
<img width="789" height="703" alt="image" src="https://github.com/user-attachments/assets/b2ad7185-abfe-41a9-816d-a2d4ca2b8d4c" />



5.  **Optimización (Iteración 2 - Ajuste Fino):**
    * Para mejorar el rendimiento, se aplicó **Ajuste Fino (Fine-Tuning)**. Se "descongelaron" las últimas 4 capas de VGG16 y se re-entrenó el modelo con una tasa de aprendizaje muy baja (`1e-5`).
Aquí podemos ver el resultado de la validación y la matriz de confusion de mejor modelo con fine tuning:


<img width="762" height="453" alt="image" src="https://github.com/user-attachments/assets/22eb6992-970a-4a97-a7b2-7c2908258d2a" />

<img width="789" height="703" alt="image" src="https://github.com/user-attachments/assets/a1294027-5731-44ae-ace5-761f0413ad64" />




---


---  
