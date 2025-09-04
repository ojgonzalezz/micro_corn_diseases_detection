#####################################################################################
# -------------------------- Data Preprocessing Utilities ---------------------------
#####################################################################################


#########################
# ---- Depdendencies ----
#########################

import pathlib
import shutil
import random
from tqdm import tqdm
import collections
import numpy as np
from PIL import Image
from adapters.data_loader import load_raw_data

####################################
# ---- Split and balance datast ----
####################################


def split_and_balance_dataset(split_ratios: tuple = (0.7, 0.15, 0.15), balanced: bool = True):
    """
    Realiza una división estratificada de un dataset de imágenes.

    Args:
        base_path (pathlib.Path): La ruta del directorio que contiene las carpetas de las clases.
        split_ratios (tuple): Una tupla con los ratios de división para train, val, y test.
        balanced (bool): Si es True, balancea el dataset usando submuestreo. Si es False, usa todas las imágenes.

    Returns:
        dict: Un diccionario con los conjuntos de datos divididos ('train', 'val', 'test'), 
              donde cada conjunto es un diccionario de la forma {'clase': [lista de imágenes]}.
    """
    #if not (base_path.exists() and base_path.is_dir()):
    #    raise FileNotFoundError(f"El directorio base '{base_path}' no existe.")

    #if not np.isclose(sum(split_ratios), 1.0):
    #    raise ValueError("Los ratios de división (train, val, test) deben sumar 1.0.")

    print("\n📦 Llamando a la función de carga de datos...")
    dataset = load_raw_data()
    print("✅ Carga de datos completada.")

    if not dataset:
        raise ValueError("No se cargó ninguna imagen. Verifica las rutas y los tipos de archivo.")

    # Lógica de división
    train_set, val_set, test_set = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    final_counts = collections.defaultdict(dict)

    if balanced:
        min_class_size = min(len(images) for images in dataset.values())
        print(f"\n⚖️  Modo balanceado: Todas las clases se reducirán a {min_class_size} imágenes.")
        
        for class_name, images in dataset.items():
            random.shuffle(images)
            sampled_images = random.sample(images, min_class_size)
            
            n_train = int(min_class_size * split_ratios[0])
            n_val = int(min_class_size * split_ratios[1])
            
            train_set[class_name] = sampled_images[:n_train]
            val_set[class_name] = sampled_images[n_train : n_train + n_val]
            test_set[class_name] = sampled_images[n_train + n_val:]
            
            final_counts[class_name] = {'train': len(train_set[class_name]), 'val': len(val_set[class_name]), 'test': len(test_set[class_name])}

    else: # balanced=False, modo desbalanceado
        print("\n📈 Modo desbalanceado: Usando todas las imágenes disponibles.")

        for class_name, images in dataset.items():
            random.shuffle(images)
            total_images = len(images)
            
            n_train = int(total_images * split_ratios[0])
            n_val = int(total_images * split_ratios[1])
            
            train_set[class_name] = images[:n_train]
            val_set[class_name] = images[n_train : n_train + n_val]
            test_set[class_name] = images[n_train + n_val:]

            final_counts[class_name] = {'train': len(train_set[class_name]), 'val': len(val_set[class_name]), 'test': len(test_set[class_name])}

    # Resumen y retorno
    print("\n" + "="*60)
    print("✅ Proceso de división completado exitosamente.")
    print("="*60)
    print("📊 Resumen de la Distribución Final:")
    header = f"{'Clase':<20} | {'Train':>7} | {'Val':>7} | {'Test':>7} | {'Total':>7}"
    print(header)
    print("-" * len(header))
    
    totals = collections.defaultdict(int)
    for class_name, counts in sorted(final_counts.items()):
        total_class = sum(counts.values())
        totals['train'] += counts['train']
        totals['val'] += counts['val']
        totals['test'] += counts['test']
        print(f"{class_name:<20} | {counts['train']:>7} | {counts['val']:>7} | {counts['test']:>7} | {total_class:>7}")
    
    print("-" * len(header))
    total_all = sum(totals.values())
    print(f"{'TOTAL':<20} | {totals['train']:>7} | {totals['val']:>7} | {totals['test']:>7} | {total_all:>7}")
    print("="*60)

    return {'train': train_set, 'val': val_set, 'test': test_set}