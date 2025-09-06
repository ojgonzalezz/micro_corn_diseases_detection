#####################################################################################
# ----------------------------------- Model Trainer ---------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import tensorflow as tf
import pathlib
from datetime import datetime
import keras_tuner as kt
import numpy as np

import mlflow
import mlflow.tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pipelines.preprocess import split_and_balance_dataset
from builders.builders import ModelBuilder

#######################
# ---- Fine Tunner ----
#######################

def train(backbone_name='VGG16', split_ratios=(0.7, 0.15, 0.15), balanced=True):
    """
    Función principal para orquestar el proceso de entrenamiento y la búsqueda
    de hiperparámetros con Keras Tuner.
    
    Args:
        split_ratios (tuple): Ratios de división para train, val y test.
        balanced (bool): Si es True, balancea el dataset. Si es False, usa el dataset original.
    """
    # --- 1. CONFIGURACIÓN ---
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 4
    
    # Parámetros para la búsqueda de Keras Tuner
    MAX_TRIALS = 10  # Número total de modelos a probar
    TUNER_EPOCHS = 10 # Número de épocas para cada modelo durante la búsqueda
    
    print("Iniciando el proceso de entrenamiento y búsqueda de hiperparámetros...")
    print(f"Dataset de origen: {DATA_DIR.name}")
    print(f"Número de clases: {NUM_CLASSES}")
    
    # --- 2. CARGAR Y PREPARAR LOS DATOS ---
    print("\nCargando y preparando los datos en memoria...")
    
    # Usar la función modificada para cargar y dividir los datos en un diccionario
    raw_dataset = split_and_balance_dataset(
        split_ratios=split_ratios,
        balanced=balanced
    )

    # Convertir el diccionario de datos en listas planas para Keras Tuner
    def flatten_data(data_dict, image_size=(224, 224)):
        images = []
        labels = []
        for class_name, image_list in data_dict.items():
            for img in image_list:
                # Redimensionar la imagen a un tamaño uniforme antes de convertirla
                resized_img = img.resize(image_size)
                images.append(np.array(resized_img))
                labels.append(class_name)
    
        return np.array(images), np.array(labels)

    # Y luego, llama a la función en tu script 'train.py'
    # con el tamaño de imagen correcto.
    X_train, y_train = flatten_data(raw_dataset['train'], image_size=IMAGE_SIZE)
    X_val, y_val = flatten_data(raw_dataset['val'], image_size=IMAGE_SIZE)
    X_test, y_test = flatten_data(raw_dataset['test'], image_size=IMAGE_SIZE)

    # Codificar las etiquetas
    label_to_int = {label: i for i, label in enumerate(np.unique(y_train))}
    y_train = np.array([label_to_int[l] for l in y_train])
    y_val = np.array([label_to_int[l] for l in y_val])
    y_test = np.array([label_to_int[l] for l in y_test])
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
    
    print("✅ Datos convertidos a tensores de NumPy.")
    
    # --- 3. INICIALIZAR EL MODEL BUILDER E INSTANCIAR EL TUNER ---
    print("\nInicializando el constructor de modelos para el Tuner...")
    
    hypermodel = ModelBuilder(
        backbone_name=backbone_name,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=NUM_CLASSES
    )

    # Configurar el directorio para guardar los checkpoints del tuner
    tuner_dir = PROJECT_ROOT / 'models' / 'tuner_checkpoints'
    tuner_dir.mkdir(parents=True, exist_ok=True)
    
    tuner = kt.RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=MAX_TRIALS,
        directory=tuner_dir,
        project_name='image_classification'
    )

    tuner.search_space_summary()
    
    # --- 4. EJECUTAR LA BÚSQUEDA DE HIPERPARÁMETROS ---
    print("\n" + "="*70)
    print("🚀 ¡Comenzando la búsqueda de hiperparámetros!")
    print("="*70)

    # La búsqueda se realiza directamente con los tensores
    tuner.search(
        x=X_train,
        y=y_train,
        epochs=TUNER_EPOCHS,
        validation_data=(X_val, y_val)
    )

    print("\n" + "="*70)
    print("✅ ¡Búsqueda de hiperparámetros completada exitosamente!")
    print("="*70)
    
    # --- 5. OBTENER EL MEJOR MODELO ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    
    print(f"\n🏆 El mejor modelo se encontró con los siguientes hiperparámetros:")
    for hp_name, value in best_hps.values.items():
        print(f"  - {hp_name}: {value}")
        
    # --- 6. EVALUAR Y GUARDAR EL MEJOR MODELO ---
    print("\n" + "="*70)
    print("📊 Evaluando el mejor modelo...")
    print("="*70)
    
    test_loss, test_acc = best_model.evaluate(x=X_test, y=y_test)
    print(f"\n✅ Precisión en el conjunto de prueba: {test_acc:.4f}")
    
    best_model_path = tuner_dir / 'best_model.keras'
    best_model.save(best_model_path)
    print(f"💾 El mejor modelo final se ha guardado en: {best_model_path}")
    
    # Devolver el tuner y los datos para análisis posterior
    return tuner, (X_test, y_test)

#if __name__ == '__main__':
#    # Llamar a la función de entrenamiento con los parámetros deseados
#    tuner_instance, test_data_sets = train(
#        split_ratios=(0.7, 0.15, 0.15),
#        balanced=True
#    )
    
    # Ahora puedes usar tuner_instance para acceder a la historia
    # Por ejemplo:
    # results = tuner_instance.get_best_trials(num_trials=1)
    # history = results[0].metrics.get_history()
    # print(history)