# Corn Diseases Detection 🌽

Este proyecto es una aplicación web impulsada por **Inteligencia Artificial** diseñada para la **detección y clasificación de enfermedades en hojas de maíz**. Permite a los usuarios subir una imagen de una hoja y recibir un diagnóstico instantáneo con un alto grado de confianza.

---

## Características principales 

* **Detección de Enfermedades**: Utiliza un modelo de aprendizaje automático para identificar enfermedades comunes como la Roya, Tizón y Mancha Gris.
* **Diagnóstico con Confianza**: Proporciona el nombre de la enfermedad detectada junto con un porcentaje de confianza para el diagnóstico.
* **Análisis Detallado**: Muestra las probabilidades de que la hoja pertenezca a otras categorías, incluyendo un estado "Saludable", para un análisis más completo.
* **Interfaz Intuitiva**: La interfaz de usuario es sencilla y fácil de usar, permitiendo subir imágenes mediante arrastrar y soltar.
* **Historial de Predicciones**: Guarda un registro de los diagnósticos previos para consulta.

---

## Estructura del repositorio

El proyecto está organizado en las siguientes carpetas y archivos clave:

* `corn-diseases-detection-api/`: Contiene el código de la API utilizada para el modelo de predicción.
* `eda/` y `eda_balanced/`: Directorios para el **Análisis Exploratorio de Datos (EDA)**.
* `models/`: Almacena los modelos de IA entrenados.
* `preprocessing/`: Archivos para el preprocesamiento de los datos y el ajuste del modelo.
* `src/`: Directorio principal con el código fuente del proyecto.
* `index.html`: El archivo principal de la interfaz de usuario web.
* `requirements.txt`: Lista las dependencias de Python necesarias para el proyecto.
* `README.md`: Este archivo.

---

## Demo y despliegue 🚀

La aplicación está desplegada y puede ser probada en vivo a través de una URL de AWS S3. La demo permite a los usuarios experimentar la funcionalidad completa del clasificador de enfermedades.

* **Repositorio del Proyecto**: [https://github.com/ojgonzalezz/micro_corn_diseases_detection](https://github.com/ojgonzalezz/micro_corn_diseases_detection)
* **URL de la Aplicación (Demo)**: `http://corn-disease-classification-2025.s3-website-us-east-1.amazonaws.com`