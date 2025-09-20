# Corn Diseases Detection 🌽

Este proyecto es una aplicación web diseñada para la **detección y clasificación de enfermedades en hojas de maíz**. Permite a los usuarios subir una imagen de una hoja y recibir un diagnóstico instantáneo con un alto grado de confianza.

-----

## Características principales

  * **Detección de enfermedades**: Utiliza un modelo de aprendizaje automático para identificar enfermedades comunes como la Roya, Tizón y Mancha Gris.
  * **Diagnóstico con confianza**: Proporciona el nombre de la enfermedad detectada junto con un porcentaje de confianza para el diagnóstico.
  * **Análisis detallado**: Muestra las probabilidades de que la hoja pertenezca a otras categorías, incluyendo un estado "Saludable", para un análisis más completo.
  * **Interfaz intuitiva**: La interfaz de usuario es sencilla y fácil de usar, permitiendo subir imágenes mediante arrastrar y soltar.
  * **Historial de predicciones**: Guarda un registro de los diagnósticos previos para consulta.

-----

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

-----

## Demo y despliegue 🚀

La aplicación está desplegada y puede ser probada en vivo a través de una URL de AWS S3. La demo permite a los usuarios experimentar la funcionalidad completa del clasificador de enfermedades.

  * **Repositorio del Proyecto**: [https://github.com/ojgonzalezz/micro\_corn\_diseases\_detection](https://github.com/ojgonzalezz/micro_corn_diseases_detection)
  * **URL de la Aplicación (Demo)**: `http://corn-disease-classification-2025.s3-website-us-east-1.amazonaws.com`

-----

## Manual de instalación y despliegue en AWS S3

Este manual explica cómo desplegar la aplicación web de detección de enfermedades del maíz directamente desde el repositorio de GitHub a un bucket de Amazon S3, permitiendo que la demo sea accesible a través de una URL pública.

### 1\. Configurar un bucket de Amazon S3

Primero, necesitas crear un **bucket de S3** que alojará los archivos estáticos de la aplicación.

  * Inicia sesión en la **Consola de Gestión de AWS**.
  * Navega al servicio **S3**.
  * Haz clic en "Crear bucket".
  * Asigna un **nombre único** al bucket. Debe ser **único a nivel global**, por lo que te sugerimos usar una convención como `proyecto-2025-demo-maiz`.
  * Elige una **región de AWS**. Generalmente, es mejor elegir la más cercana a tus usuarios para reducir la latencia.
  * En "Opciones de configuración de objeto", desmarca la opción "**Bloquear todo el acceso público**" (Block all public access) y confirma que estás haciendo público el bucket. Esto es crucial para que la aplicación web sea accesible.
  * Haz clic en "Crear bucket".

### 2\. Sube los archivos del proyecto a tu bucket de S3

Ahora, debes transferir los archivos del repositorio a tu bucket recién creado.

  * Descarga el proyecto de GitHub. Puedes hacerlo con `git clone https://github.com/ojgonzalezz/micro_corn_diseases_detection` o descargando el archivo ZIP del repositorio.
  * Dentro del repositorio, la **interfaz de usuario** está en el archivo `index.html`. Debes subir este archivo y sus dependencias (si las hubiera, como archivos CSS o JS) al **directorio raíz** de tu bucket. En este proyecto, el `index.html` es el único archivo de interfaz.
  * En la consola de S3, navega a tu bucket.
  * Haz clic en "Cargar" (Upload).
  * Arrastra y suelta el archivo `index.html` del proyecto en el área de carga.
  * Haz clic en "Cargar" nuevamente para iniciar la subida.

### 3\. Habilita el alojamiento de sitios web estáticos

Para que el bucket funcione como un sitio web, debes habilitar esta característica.

  * En la consola de S3, selecciona tu bucket.
  * Ve a la pestaña "**Propiedades**" (Properties).
  * Baja hasta la sección "**Alojamiento de sitios web estáticos**" (Static website hosting) y haz clic en "Editar".
  * Selecciona "Habilitar" (Enable).
  * En "Documento de índice", escribe `index.html`. Este es el archivo que se cargará por defecto cuando alguien acceda a la URL del sitio.
  * Haz clic en "Guardar cambios".

### 4\. Actualiza la política del bucket para acceso público

Aunque desmarcaste el bloqueo de acceso público, necesitas una **política de bucket** explícita para permitir que los usuarios vean los archivos.

  * En la consola de S3, selecciona tu bucket y ve a la pestaña "**Permisos**" (Permissions).
  * En la sección "**Política de bucket**" (Bucket policy), haz clic en "Editar".
  * Copia y pega la siguiente política, asegurándote de reemplazar `YOUR-BUCKET-NAME` con el nombre exacto de tu bucket. Esta política permite que cualquier persona (el `*`) obtenga objetos (`s3:GetObject`) de tu bucket.

<!-- end list -->

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
        }
    ]
}
```

  * Haz clic en "Guardar cambios".

### 5\. Accede a tu aplicación web

¡Ya está\! Ahora puedes acceder a tu aplicación web a través de la URL de alojamiento de sitios web estáticos.

  * Regresa a la pestaña "**Propiedades**" de tu bucket.
  * Baja hasta la sección "**Alojamiento de sitios web estáticos**".
  * Encontrarás la **URL del punto de enlace del bucket** (Bucket website endpoint). Esta es la dirección que debes usar para acceder a la aplicación web.

Cuando un usuario visite esa URL, AWS S3 servirá el archivo `index.html` que subiste, permitiendo el uso de la aplicación de detección de enfermedades del maíz.
