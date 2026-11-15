
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Definir rutas de entrada y salida
# --------------------------------------------------
RUTA_O3 = Path("outputs/tables/o3_dataset_particionado.csv")
RUTA_METEO = Path("outputs/tables/london_historical_weather.csv")
RUTA_SALIDA = Path("outputs/tables/o3_dataset_meteorologia.csv")

# Variables meteorológicas que incorporaremos al modelo
VARIABLES_METEO = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
]

# --------------------------------------------------
# 2. Cargar ambos conjuntos de datos
# --------------------------------------------------
o3 = pd.read_csv(RUTA_O3)
meteo = pd.read_csv(RUTA_METEO)

# Convertir las fechas a datetime con zona horaria UTC
o3["fecha_utc"] = pd.to_datetime(o3["fecha_utc"], utc=True)
meteo["fecha_utc"] = pd.to_datetime(meteo["fecha_utc"], utc=True)

# Comprobar que todas las columnas necesarias existen
faltantes = [c for c in VARIABLES_METEO if c not in meteo.columns]
if faltantes:
    raise ValueError(f"Faltan variables meteorológicas: {faltantes}")

# Comprobar que cada fecha aparece una sola vez en cada tabla
if o3["fecha_utc"].duplicated().any():
    raise ValueError("Hay fechas duplicadas en el conjunto de ozono.")

if meteo["fecha_utc"].duplicated().any():
    raise ValueError("Hay fechas duplicadas en el conjunto meteorológico.")

# --------------------------------------------------
# 3. Preparar la tabla meteorológica
# --------------------------------------------------
# Seleccionar únicamente fecha y variables para la unión
meteo_modelo = meteo[["fecha_utc"] + VARIABLES_METEO].copy()

# Codificar la dirección del viento de forma circular.
# Así, 359° y 1° se representan como direcciones cercanas.
angulo = np.deg2rad(meteo_modelo["wind_direction_10m"])
meteo_modelo["wind_direction_sin"] = np.sin(angulo)
meteo_modelo["wind_direction_cos"] = np.cos(angulo)

# --------------------------------------------------
# 4. Unir usando la cuadrícula horaria de ozono
# --------------------------------------------------
# left conserva todas las filas de ozono, incluso si falta
# alguna observación meteorológica en una hora determinada.
df = o3.merge(
    meteo_modelo,
    on="fecha_utc",
    how="left",
    validate="one_to_one",
    indicator=True,
)

# Registrar si cada hora encontró datos meteorológicos
df["meteorologia_encontrada"] = df["_merge"].eq("both")
df = df.drop(columns="_merge")

# Ordenar cronológicamente y comprobar la cuadrícula
df = df.sort_values("fecha_utc").reset_index(drop=True)

# --------------------------------------------------
# 5. Validar la integración
# --------------------------------------------------
print("\n=== VALIDACIÓN DE LA INTEGRACIÓN ===")
print("Filas del conjunto de ozono:", len(o3))
print("Filas después de unir:", len(df))
print("Fechas duplicadas:", df["fecha_utc"].duplicated().sum())
print("Horas con meteorología:", df["meteorologia_encontrada"].sum())
print("Horas sin meteorología:", (~df["meteorologia_encontrada"]).sum())
print("Inicio UTC:", df["fecha_utc"].min())
print("Fin UTC:", df["fecha_utc"].max())

print("\nAusentes por variable meteorológica:")
print(df[VARIABLES_METEO].isna().sum().to_string())

# Comprobar que no se alteraron las etiquetas de partición
if "particion" in o3.columns:
    print("\nDistribución de particiones:")
    print(df["particion"].value_counts(dropna=False).to_string())

# Verificar que el objetivo original se conserva
if "valor_o3_modelo" in o3.columns:
    assert df["valor_o3_modelo"].equals(o3.sort_values("fecha_utc")
                                        .reset_index(drop=True)["valor_o3_modelo"])

# --------------------------------------------------
# 6. Guardar el conjunto integrado
# --------------------------------------------------
RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(RUTA_SALIDA, index=False)

print("\nArchivo guardado en:", RUTA_SALIDA)
print("Columnas disponibles:", df.columns.tolist())
