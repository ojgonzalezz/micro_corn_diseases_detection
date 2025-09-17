# Archivo: src/train.py

import tensorflow as tf
import pathlib
from datetime import datetime
import mlflow
import mlflow.tensorflow

# Importar las funciones que ya creamos en los otros archivos
from data_pipeline import create_data_generators
from model import build_model
# Importar los Callbacks necesarios
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import mlflow.tensorflow

def train():
    
    mlflow.tensorflow.autolog()

    """
    Función principal para orquestar el proceso de entrenamiento del modelo,
    incluyendo callbacks para un entrenamiento robusto.
    """
    # --- 1. CONFIGURACIÓN ---
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    SPLIT_DATA_DIR = PROJECT_ROOT / 'dataset_split_balanced'
    
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32 
    NUM_CLASSES = 4
    EPOCHS = 30

   # ---------- MLflow: tracking + experiment ----------
    # Usa una carpeta persistente en tu Drive
    MLFLOW_DIR = PROJECT_ROOT / "mlruns"
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")            
    mlflow.set_experiment("vgg16_baseline")                  
    mlflow.tensorflow.autolog()                              
    # ----------------------------------------------------

    print("Iniciando el proceso de entrenamiento...")
    print(f"Dataset: {SPLIT_DATA_DIR.name}")
    print(f"Épocas: {EPOCHS}, Tamaño de Lote: {BATCH_SIZE}")

    # --- 2. PREPARAR LOS DATOS ---
    print("\nCargando y preparando los datos...")
    train_generator, validation_generator, test_generator = create_data_generators(
        base_dir=SPLIT_DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # --- 3. CONSTRUIR EL MODELO ---
    print("\nConstruyendo la arquitectura del modelo...")
    model = build_model(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=NUM_CLASSES
    )
    model.summary() # Puedes descomentar esto si quieres ver el resumen cada vez

    # --- 4. CONFIGURAR CALLBACKS ---
    print("\nConfigurando Callbacks...")
    
    # Crear carpeta para guardar los modelos si no existe
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardará el mejor modelo basado en la precisión de validación
    checkpoint_cb = ModelCheckpoint(
        filepath=models_dir / 'best_model.keras', # Guardará el mejor modelo aquí
        save_best_only=True,                     # Solo guarda si el modelo mejora
        monitor='val_accuracy',                  # Métrica a monitorear
        mode='max',                              # Queremos maximizar la precisión
        verbose=1                                # Imprime un mensaje cuando guarda
    )

    # Detendrá el entrenamiento si no hay mejora después de 3 épocas
    early_stopping_cb = EarlyStopping(
        monitor='val_accuracy', # Métrica a monitorear
        patience=3,             # Número de épocas a esperar sin mejora
        restore_best_weights=True # Restaura los pesos del mejor modelo al finalizar
    )
    
    print("Callbacks 'ModelCheckpoint' y 'EarlyStopping' listos.")

    # --- 5. ENTRENAR EL MODELO ---
    print("\n" + "="*70)
    print("🚀 ¡Comenzando el entrenamiento!")
    print("="*70)

    with mlflow.start_run(run_name="vgg16_baseline"):
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            callbacks=[checkpoint_cb, early_stopping_cb] # <-- AQUÍ SE AÑADEN LOS CALLBACKS
        )

        # Guardar el modelo final como artefacto de MLflow
        mlflow.keras.log_model(model, "final_model")

        # También puedes registrar hiperparámetros o métricas manualmente si quieres
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("image_size", IMAGE_SIZE)

    print("\n" + "="*70)
    print("✅ ¡Entrenamiento completado exitosamente!")
    print("="*70)

    # --- 6. GUARDAR EL MODELO FINAL (OPCIONAL, YA QUE CHECKPOINT GUARDA EL MEJOR) ---
    # Es una buena práctica guardar también el modelo final para comparar.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_model_path = models_dir / f'final_model_{timestamp}.keras'
    model.save(final_model_path)
    print(f"💾 Modelo final guardado en: {final_model_path}")
    print(f"🏆 El mejor modelo se guardó automáticamente en: {models_dir / 'best_model.keras'}")

    return history, model, test_generator

if __name__ == '__main__':
    train_history, trained_model, test_data = train()