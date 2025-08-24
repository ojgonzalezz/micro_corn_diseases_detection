# 🌽 Análisis Exploratorio de Datos
### Detección de Enfermedades en Cultivos de Maíz

Este documento presenta los hallazgos de la primera fase del proyecto, centrada en el análisis y la comprensión de los datos iniciales.

---

## 📜 Problema y Contexto
Las enfermedades del maíz, como la roya común, el tizón foliar y la mancha gris, representan una amenaza crítica para la seguridad alimentaria. El diagnóstico tradicional mediante inspección visual es un proceso lento, subjetivo y dependiente de la pericia del observador. Este proyecto busca validar la viabilidad de un sistema de diagnóstico automatizado mediante Inteligencia Artificial para superar estas limitaciones.

---

## 📊 Dataset Inicial
Para el análisis, se utilizó el dataset público **"Corn or Maize Leaf Disease Dataset"** de Kaggle, una compilación de imágenes de las fuentes PlantVillage y PlantDoc.

* **Total de Imágenes:** 4,188
* **Formato:** JPEG (.jpg)
* **Distribución de Clases (Inicial):**
    * Roya Común (Common Rust): 1,306 imágenes (31.2%)
    * Sana (Healthy): 1,162 imágenes (27.7%)
    * Tizón (Blight): 1,146 imágenes (27.4%)
    * Mancha Gris (Gray Leaf Spot): 574 imágenes (13.7%)

> **Observación Clave:** El dataset inicial presenta un notable desbalance, con la clase **"Mancha Gris"** significativamente subrepresentada. Este hallazgo es fundamental para las siguientes etapas del proyecto.

---

## 🔬 Hallazgos del Análisis Exploratorio de Datos (EDA)

### Validación e Integridad de Datos
Se realizó una validación estructural del dataset para confirmar la cantidad de clases, el número de imágenes y la integridad de los archivos. Se encontró y corrigió una inconsistencia de formato (un archivo **.jpeg** en lugar de **.jpg**) en la clase "Blight", asegurando la homogeneidad del conjunto de datos.

### Análisis Cualitativo Visual
La inspección de muestras aleatorias reveló una buena calidad de imagen general (nitidez y enfoque). Se destacó una alta variabilidad en iluminación, escala y ángulos de captura, lo cual es beneficioso para entrenar un modelo más robusto y generalizable.

> **Desafío Principal Identificado:** Se observó una alta similitud morfológica entre las lesiones en etapas avanzadas de **"Mancha Gris"** y **"Tizón"**, lo que anticipa el principal reto de clasificación para el modelo de IA.

### Análisis Cuantitativo de Características Físicas
* **Dimensiones:** Se confirmó una considerable variabilidad en el tamaño (alto y ancho) de las imágenes, lo que fundamenta la necesidad de un paso de redimensionamiento estándar antes de alimentar el modelo.
* **Distribución de Color:** El análisis de histogramas de color, particularmente en el canal verde, demostró ser un rasgo altamente discriminatorio. Las hojas sanas ("Healthy") mostraron un perfil de color verde único y vibrante, claramente distinto al de las hojas enfermas. Esto valida el potencial del color como una característica potente para la clasificación automática y justifica la necesidad de normalizar los valores de los píxeles.