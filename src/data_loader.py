import os
import sys
import pathlib
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join("..", "src")))

# ---- Data loader ----
def load_raw_data():
    # Importaciones movidas dentro de la función
    from src.path_finder import ProjectPaths
    from src.utils import load_images_from_folder
    
    pp = ProjectPaths()
    data_paths = pp.get_structure()
    
    raw_data = {}
    for category, path in data_paths['raw']['data'].items():
        raw_data[category] = load_images_from_folder(path)
    print("✅ Las imágenes de todas las categorías se han cargado exitosamente")
    return raw_data