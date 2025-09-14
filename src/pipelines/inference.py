#####################################################################################
# -------------------------------- Inference Pipeline -------------------------------
#####################################################################################

########################
# ---- Dependencies ----
########################

import os
import pathlib
import sys
from core.load_env import EnvLoader
from core.path_finder import ProjectPaths
import numpy as np
from PIL import Image
import tensorflow as tf

project_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

##############################
# ---- Inference Pipeline ----
##############################

class Inference:

    def __init__(self):
        project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))

        self.env_vars = EnvLoader().get_all()
        model_paths_obj = ProjectPaths(data_subpath=("models",)).get_structure()
        self.model_dir = model_paths_obj["exported"]
        self.model_path = pathlib.Path(self.model_dir) / "best_VGG16.keras"

        self.image_size = (
            int(self.env_vars.get("IMAGE_WIDTH", 224)),
            int(self.env_vars.get("IMAGE_HEIGHT", 224))
        )
        self.class_names = self.env_vars.get("CLASS_NAMES", "").split(',')
        
        if not self.model_path.exists():
            print(f"❌ Error: Modelo no encontrado en '{self.model_path}'. Asegúrate de que el entrenamiento se completó y el modelo fue guardado.")
            self.model = None
        else:
            print(f"🚀 Cargando modelo desde: {self.model_path}...")
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print("✅ Modelo cargado exitosamente.")
                self.model.summary()
            except Exception as e:
                print(f"❌ Error al cargar el modelo: {e}")
                self.model = None

    def predict_on_image(self, image_path: str) -> dict:
        """
        Genera una predicción para una imagen, incluyendo la confianza de todas las clases.
        
        Args:
            image_path (str): La ruta a la imagen de entrada.

        Returns:
            dict: Un diccionario con la predicción principal y un desglose completo de la confianza por clase.
        """
        if self.model is None:
            return {"prediction": "Error", "confidence": 0.0, "all_confidences": {}, "message": "Modelo no cargado"}
        
        image_path_obj = pathlib.Path(image_path)
        if not image_path_obj.exists():
            print(f"❌ Error: La imagen en '{image_path}' no fue encontrada.")
            return {"prediction": "Error", "confidence": 0.0, "all_confidences": {}, "message": "Imagen no encontrada"}

        print(f"🖼️ Cargando y preprocesando la imagen de '{image_path}'...")
        try:
            image = Image.open(image_path_obj).resize(self.image_size)
            image_array = np.array(image).astype('float32') / 255.0
            image_array = np.expand_dims(image_array, axis=0)
        except Exception as e:
            print(f"❌ Error al preprocesar la imagen: {e}")
            return {"prediction": "Error", "confidence": 0.0, "all_confidences": {}, "message": f"Error de preprocesamiento: {e}"}

        print("🔮 Generando predicción...")
        predictions = self.model.predict(image_array)
        
        # Crear el diccionario con todas las clases y sus confianzas
        all_confidences = {
            class_name: float(confidence)
            for class_name, confidence in zip(self.class_names, predictions[0])
        }
        
        # Obtener el índice de la clase con la mayor probabilidad
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_index]
        confidence = all_confidences[predicted_class]
        
        print(f"\n✅ Predicción principal: {predicted_class} con una confianza del {confidence:.2%}")
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "all_confidences": all_confidences,
            "message": "Predicción exitosa"
        }

