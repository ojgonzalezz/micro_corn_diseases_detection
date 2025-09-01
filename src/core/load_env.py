import os
from dotenv import load_dotenv, dotenv_values
from typing import Optional, Dict

class EnvLoader:
    def __init__(self, env_file: Optional[str] = None):
        # Por defecto: src/core/.env (archivo junto a este módulo)
        if env_file is None:
            env_file = os.path.join(os.path.dirname(__file__), ".env")

        # Hacemos absoluta la ruta del .env
        env_file = os.path.abspath(env_file)
        if not os.path.exists(env_file):
            raise FileNotFoundError(f"No se encontró el archivo .env en {env_file}")

        self.env_file = env_file
        self.env_dir = os.path.dirname(self.env_file)  # src/core
        self.project_root = os.path.abspath(
            os.path.join(self.env_dir, os.pardir, os.pardir)
        )

        # Cargar las variables en el entorno
        load_dotenv(self.env_file)
        # También las guardamos en un dict para acceso directo
        self._env_vars = dotenv_values(self.env_file)

    def _clean(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = value.strip()
        # quitar comillas accidentales
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1].strip()
        # expandir ~ y variables de entorno si existen
        v = os.path.expanduser(os.path.expandvars(v))
        return v or None

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        raw = os.getenv(key, default)
        return self._clean(raw)

    def get_all(self) -> Dict[str, Optional[str]]:
        """
        Devuelve todas las variables definidas en el archivo .env
        en forma de diccionario.
        """
        return {k: self._clean(v) for k, v in self._env_vars.items()}

    def get_path(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Devuelve la ruta absoluta (intentando varias bases) y muestra info:
        - Si es directorio: imprime cuántos archivos y carpetas contiene y un preview.
        - Si es archivo: lo indica.
        - Si no existe: muestra advertencia y las rutas que intentó.
        """
        raw = self.get(key, default)
        if not raw:
            print(f"[WARN] La variable {key} no está definida en el .env")
            return None

        candidates = []
        if os.path.isabs(raw):
            candidates = [os.path.normpath(raw)]
        else:
            candidates.append(os.path.normpath(os.path.join(self.env_dir, raw)))        # relativo al .env
            candidates.append(os.path.normpath(os.path.join(self.project_root, raw)))   # relativo al root
            candidates.append(os.path.normpath(os.path.join(os.getcwd(), raw)))         # relativo al cwd
            candidates.append(os.path.normpath(raw))                                    # tal cual

        chosen = None
        for c in candidates:
            if os.path.exists(c):
                chosen = os.path.abspath(c)
                break

        if not chosen:
            chosen = os.path.abspath(candidates[1]) if len(candidates) > 1 else os.path.abspath(candidates[0])

        if os.path.isdir(chosen):
            entries = sorted(os.listdir(chosen))
            files = [e for e in entries if os.path.isfile(os.path.join(chosen, e))]
            dirs = [e for e in entries if os.path.isdir(os.path.join(chosen, e))]
            print(f"[INFO] {key} -> {chosen}")
            print(f"       Contiene {len(files)} archivos y {len(dirs)} carpetas.")
            if entries:
                print(f"       Primeros elementos: {entries[:5]}")
        elif os.path.isfile(chosen):
            print(f"[INFO] {key} -> {chosen} (es un archivo válido)")
        else:
            print(f"[WARN] La ruta {chosen} NO existe. Se intentaron las siguientes rutas (en orden):")
            for i, c in enumerate(candidates, start=1):
                print(f"   {i}. {c}")

        return chosen
