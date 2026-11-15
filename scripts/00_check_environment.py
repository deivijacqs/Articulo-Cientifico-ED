"""Comprueba que el entorno y los archivos básicos del proyecto están disponibles."""
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = ["numpy", "pandas", "requests", "yaml", "dotenv", "sklearn", "torch"]

def main():
    print("Python:", sys.version.split()[0])
    print("Raíz del proyecto:", ROOT)
    missing = []
    for module in REQUIRED_MODULES:
        try:
            imported = importlib.import_module(module)
            print(f"[OK] {module}: {getattr(imported, '__version__', 'instalado')}")
        except Exception as exc:
            missing.append(module)
            print(f"[ERROR] {module}: {exc}")
    for folder in ["data/raw/openaq", "data/processed", "data/external/era5",
                   "outputs/figures", "outputs/tables", "logs", "models"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    if missing:
        raise SystemExit("Faltan dependencias. Activa el entorno e instala environment.yml.")
    print("\\nEntorno verificado. Puedes continuar con el script 01.")

if __name__ == "__main__":
    main()
