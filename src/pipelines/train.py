#####################################################################################
# ----------------------------------- Model Trainer ---------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import tensorflow as tf
import pathlib
import numpy as np
import mlflow
import mlflow.keras as mlflow_keras
import keras_tuner as kt
from tensorflow.keras.callbacks import Callback
import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Asegúrate de que las rutas de importación sean correctas para tu proyecto
from pipelines.preprocess import split_and_balance_dataset
from builders.builders import ModelBuilder

##########################
# ---- MLflow Callbacks ----
##########################

class MLflowKerasTunerCallback(Callback):
    """
    Callback personalizado para MLflow que registra cada 'trial' de Keras Tuner
    como un 'run' individual de MLflow, incluyendo parámetros y métricas por epoch.
    """
    def __init__(self, tuner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tuner = tuner
        self.mlflow_runs = {}
        self.current_trial_id = None

    def on_trial_begin(self, trial):
        # Inicia un nuevo run en MLflow (anidado dentro del experimento)
        run = mlflow.start_run(nested=True, run_name=f"trial-{trial.trial_id}")
        self.mlflow_runs[trial.trial_id] = run
        self.current_trial_id = trial.trial_id

        # Loggea los hiperparámetros del trial en MLflow
        for hp_name, hp_value in trial.hyperparameters.values.items():
            mlflow.log_param(hp_name, hp_value)

    def on_epoch_end(self, epoch, logs=None):
        """
        Registra métricas por epoch (accuracy, loss, val_accuracy, val_loss, etc.)
        """
        if logs:
            for metric_name, metric_value in logs.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)

    def on_trial_end(self, trial):
        """
        Cierra el run de MLflow al finalizar el trial
        """
        mlflow.end_run()
        
        self.current_trial_id = None


#######################
# ---- Fine Tunner ----
#######################

def train(backbone_name='VGG16', split_ratios=(0.7, 0.15, 0.15), balanced=True):
    """
    Función principal para orquestar el proceso de entrenamiento y la búsqueda
    de hiperparámetros con Keras Tuner, rastreando los experimentos con MLflow.
    
    Args:
        backbone_name (str): Nombre del modelo base a usar (ej. 'VGG16', 'ResNet50').
        split_ratios (tuple): Ratios de división para train, val y test.
        balanced (bool): Si es True, balancea el dataset. Si es False, usa el dataset original.
    
    Returns:
        kt.Tuner: El objeto tuner con los resultados de la búsqueda.
        tuple: Una tupla con los datos de prueba (X_test, y_test).
    """
    # --- 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
    MLRUNS_PATH = os.path.join(os.path.dirname(os.getcwd()), 'models', 'mlruns')
    print('mlruns directory =', MLRUNS_PATH)
    os.makedirs(MLRUNS_PATH, exist_ok=True)

    # Tracking URI (usar formato file:/// con / en lugar de \)
    mlruns_uri = f"file:///{os.path.abspath(MLRUNS_PATH).replace(os.sep, '/')}"
    mlflow.set_tracking_uri(mlruns_uri)

    print("📂 Tracking URI actual:", mlflow.get_tracking_uri())

    # Asegurar que el experimento existe
    experiment_name = "image_classification_experiment"
    mlflow.set_experiment(experiment_name)

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    
    # Parámetros para la búsqueda de Keras Tuner
    MAX_TRIALS = 10  # Número total de modelos a probar
    TUNER_EPOCHS = 10 # Número de épocas para cada modelo durante la búsqueda
    FACTOR = 3     # Factor de reducción para el algoritmo Hyperband.
    MAX_EPOCHS = 20 # Número máximo de épocas para cualquier modelo.


    print("Iniciando el proceso de entrenamiento y búsqueda de hiperparámetros con Hyperband...")
    print(f"Dataset de origen: {DATA_DIR.name}")
    print(f"Número de clases: {NUM_CLASSES}")

    # --- 2. CARGAR Y PREPARAR LOS DATOS ---
    print("\n📦 Cargando y preparando los datos en memoria...")
    
    raw_dataset = split_and_balance_dataset(
        split_ratios=split_ratios,
        balanced=balanced
    )

    def flatten_data(data_dict, image_size=(224, 224)):
        images = []
        labels = []
        for class_name, image_list in data_dict.items():
            for img in image_list:
                resized_img = img.resize(image_size)
                images.append(np.array(resized_img))
                labels.append(class_name)
        
        return np.array(images), np.array(labels)

    X_train, y_train = flatten_data(raw_dataset['train'], image_size=IMAGE_SIZE)
    X_val, y_val = flatten_data(raw_dataset['val'], image_size=IMAGE_SIZE)
    X_test, y_test = flatten_data(raw_dataset['test'], image_size=IMAGE_SIZE)

    # Después de la función flatten_data()
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("Error: Los datos de entrenamiento contienen valores no válidos.")
        exit()
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        print("Error: Los datos de entrenamiento contienen valores no válidos.")
        exit()
    if np.isnan(X_test).any() or np.isinf(X_test).any():
        print("Error: Los datos de entrenamiento contienen valores no válidos.")
        exit()

    label_to_int = {label: i for i, label in enumerate(np.unique(y_train))}
    y_train = np.array([label_to_int[l] for l in y_train])
    y_val = np.array([label_to_int[l] for l in y_val])
    y_test = np.array([label_to_int[l] for l in y_test])
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
    
    print("✅ Datos convertidos a tensores de NumPy.")
    
    # --- 3. INICIALIZAR EL MODEL BUILDER E INSTANCIAR EL TUNER ---
    print("\n🛠️  Inicializando el constructor de modelos para el Tuner...")
    
    hypermodel = ModelBuilder(
        backbone_name=backbone_name,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=NUM_CLASSES
    )

    tuner_dir = PROJECT_ROOT / 'models' / 'tuner_checkpoints'
    tuner_dir.mkdir(parents=True, exist_ok=True)

    print('KERAS TUNER DIR =', tuner_dir)
    tuner = kt.Hyperband(
        hypermodel,
        objective='val_accuracy',
        max_epochs=MAX_EPOCHS,
        factor=FACTOR,
        directory=tuner_dir,
        project_name='image_classification'
    )
    

    tuner.search_space_summary()
    
    # --- 4. CONFIGURAR CALLBACKS ---
    print("\n⚙️  Configurando Callbacks para la búsqueda...")
    
    checkpoint_cb = ModelCheckpoint(
        filepath=tuner_dir / 'best_trial_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    early_stopping_cb = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    callbacks = [
        checkpoint_cb, 
        early_stopping_cb
    ]

    # --- 5. EJECUTAR LA BÚSQUEDA DE HIPERPARÁMETROS CON MLflow ---
    print("\n" + "="*70)
    print("🚀 ¡Comenzando la búsqueda de hiperparámetros con MLflow!")
    print("="*70)

    # Iniciar un run de MLflow que encapsula toda la búsqueda
    with mlflow.start_run(run_name=f"{backbone_name}_tuner_search"):
        tuner.search(
            x=X_train,
            y=y_train,
            epochs=TUNER_EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            batch_size=BATCH_SIZE
        )

        # Al terminar, log de cada trial manualmente
        for trial in tuner.oracle.trials.values():
            with mlflow.start_run(nested=True, run_name=f"trial-{trial.trial_id}"):
                for hp_name, hp_value in trial.hyperparameters.values.items():
                    mlflow.log_param(hp_name, str(hp_value))

                if trial.metrics.metrics:
                    for metric_name, metric_obj in trial.metrics.metrics.items():
                    # metric_obj.history es una lista de floats (uno por epoch)
                        history = metric_obj.get_history() if hasattr(metric_obj, "get_history") else metric_obj.history
                        if history:
                        # loggea todos los valores por epoch
                            for obs in history:
                                # obs puede ser un MetricObservation o un número
                                if hasattr(obs, "value"):
                                    val = obs.value
                                    if isinstance(val, (list, tuple)):
                                        # recorrer cada valor dentro de la lista
                                        for i, v in enumerate(val):
                                            mlflow.log_metric(metric_name, float(v), step=(getattr(obs, "step", 0) or 0) + i)
                                    else:
                                        mlflow.log_metric(
                                            metric_name,
                                            float(val),
                                            step=getattr(obs, "step", None) or 0
                                        )
                                else:
                                    # obs es ya un float o int
                                    mlflow.log_metric(metric_name, float(obs), step=history.index(obs))
    
    # --- 6. OBTENER Y GUARDAR EL MEJOR MODELO ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    #if best_hps:
    #    best_hps = best_hps[0]
    #else:
    #    print("⚠️ No se encontraron hiperparámetros óptimos, usando los iniciales por defecto")
    #    best_hps = tuner.oracle.get_space().get_hyperparameters()  # fallback

    with mlflow.start_run(run_name=f"{backbone_name}_best_model", nested=True):
        print("\n📊 Evaluando el mejor modelo en el conjunto de prueba...")
        test_loss, test_acc = best_model.evaluate(x=X_test, y=y_test)
        print(f"\n✅ Precisión en el conjunto de prueba: {test_acc:.4f}")
        mlflow.log_params(best_hps.values)
        mlflow.keras.log_model(best_model, "model")
        mlflow.log_metric("test_accuracy", test_acc)

    print(f"\n🏆 El mejor modelo se encontró con los siguientes hiperparámetros:")
    for hp_name, value in best_hps.values.items():
        print(f"   - {hp_name}: {value}")

    print("\n" + "="*70)
    print("✅ ¡Búsqueda de hiperparámetros completada exitosamente!")
    print("="*70)

    exported_model_dir = PROJECT_ROOT / 'models' / 'exported'
    exported_model_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = exported_model_dir / f'best_{backbone_name}.keras'
    best_model.save(best_model_path)
    print(f"\n💾 El mejor modelo final se ha guardado en: {best_model_path}")

    # --- 7. EVALUAR EL MEJOR MODELO EN EL CONJUNTO DE PRUEBA ---
    print("\n" + "="*70)
    print("📊 Evaluando el mejor modelo en el conjunto de prueba...")
    print("="*70)
    
    test_loss, test_acc = best_model.evaluate(x=X_test, y=y_test)
    print(f"\n✅ Precisión en el conjunto de prueba: {test_acc:.4f}")
    
    return tuner, (X_test, y_test)