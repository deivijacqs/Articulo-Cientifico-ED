
from pathlib import Path
import pandas as pd

# Define el archivo de entrada y la carpeta de salida.
RUTA_ENTRADA = Path("outputs/tables/o3_dataset_horario.csv")
CARPETA_SALIDA = Path("outputs/tables")

# Carga el conjunto horario y ordena los registros por fecha UTC.
df = pd.read_csv(RUTA_ENTRADA, parse_dates=["fecha_utc"])
df = df.sort_values("fecha_utc").reset_index(drop=True)

# Comprueba que la serie tenga suficientes filas para realizar la división.
if len(df) < 10:
    raise ValueError("No hay suficientes filas para dividir la serie.")

# Calcula los puntos de corte usando posiciones temporales, sin mezclar fechas.
n = len(df)
corte_train = int(n * 0.70)
corte_val = int(n * 0.85)

# Asigna cada fila a una partición cronológica.
df["particion"] = "test"
df.loc[:corte_train - 1, "particion"] = "train"
df.loc[corte_train:corte_val - 1, "particion"] = "validation"

# Guarda el conjunto completo con la partición asignada.
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
ruta_completa = CARPETA_SALIDA / "o3_dataset_particionado.csv"
df.to_csv(ruta_completa, index=False)

# Guarda también cada partición por separado, sin eliminar las horas ausentes.
for particion in ["train", "validation", "test"]:
    parte = df[df["particion"] == particion].copy()

    ruta_salida = CARPETA_SALIDA / f"o3_{particion}.csv"
    parte.to_csv(ruta_salida, index=False)

# Muestra las fechas de corte y la cantidad de observaciones utilizables.
print("\n=== DIVISIÓN CRONOLÓGICA DE O3 ===")

for particion in ["train", "validation", "test"]:
    parte = df[df["particion"] == particion]

    # Convierte la máscara a booleanos para contar las observaciones entrenables.
    mascara = parte["observacion_entrenable"].fillna(False).astype(bool)

    print(f"\nPartición: {particion}")
    print(f"Filas horarias: {len(parte)}")
    print(f"Inicio UTC: {parte['fecha_utc'].min()}")
    print(f"Fin UTC: {parte['fecha_utc'].max()}")
    print(f"Observaciones entrenables: {int(mascara.sum())}")

    # Calcula la proporción de observaciones entrenables dentro de la partición.
    cobertura = 100 * mascara.mean() if len(parte) else 0
    print(f"Cobertura entrenable: {cobertura:.2f}%")

print("\nArchivos generados:")
print(ruta_completa)
print(CARPETA_SALIDA / "o3_train.csv")
print(CARPETA_SALIDA / "o3_validation.csv")
print(CARPETA_SALIDA / "o3_test.csv")
