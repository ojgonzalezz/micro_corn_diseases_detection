import io
import json
import numpy as np
from PIL import Image
import tensorflow as tf

import tensorflow as tf



# --- Config ---
IMG_SIZE = (224, 224)  # ajusta al tamaño de tu entrenamiento
MODEL_PATH = "models/fine_tuned_best_model.keras"  # .keras/.h5 o SavedModel dir
LABELS_PATH = "app/labels.json"    # opcional

# --- Carga de modelo ---
_model =  tf.keras.models.load_model(MODEL_PATH)
_labels = None
try:
    with open(LABELS_PATH, "r") as f:
        _labels = json.load(f)  # p.ej. ["healthy", "roya", "tizon", "mancha_gris"]
except FileNotFoundError:
    pass

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def predict(file_bytes: bytes):
    x = preprocess_image(file_bytes)
    preds = _model.predict(x, verbose=0)[0]  # shape (num_classes,)
    probs = preds.astype(float).tolist()
    idx = int(np.argmax(preds))
    label = _labels[idx] if _labels else str(idx)
    
    # Crear un diccionario con todas las probabilidades por clase
    class_probs = {}
    if _labels:
        for i, class_label in enumerate(_labels):
            class_probs[class_label] = probs[i]
    else:
        for i, prob in enumerate(probs):
            class_probs[str(i)] = prob
    
    return {
        "predicted_label": label,
        "predicted_index": idx,
        "confidence": probs[idx],  # probabilidad de la clase predicha
        "all_probabilities": class_probs,  # todas las probabilidades
        "raw_probabilities": probs  # array original de probabilidades
    }

