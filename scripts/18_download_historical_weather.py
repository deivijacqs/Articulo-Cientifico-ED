
from pathlib import Path
import json
import requests
import pandas as pd

# Coordenadas registradas para la estación Bexley - Belvedere West.
LATITUD = 51.494649
LONGITUD = 0.137279

# Periodo que coincide con el conjunto histórico de ozono.
FECHA_INICIO = "2021-04-28"
FECHA_FIN = "2023-12-31"

# Variables meteorológicas horarias que pueden ayudar a estudiar el ozono.
VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
]

# Selecciona el producto de reanálisis para mantener una fuente consistente.
MODELO = "era5"  # Reanálisis con datos disponibles para las ocho variables

# Define las rutas de salida para conservar tanto la respuesta original como el CSV.
CARPETA_RAW = Path("data/raw/openaq_historical_weather")
CARPETA_TABLAS = Path("outputs/tables")

CARPETA_RAW.mkdir(parents=True, exist_ok=True)
CARPETA_TABLAS.mkdir(parents=True, exist_ok=True)

# Construye la petición a la API de archivo histórico.
url = "https://archive-api.open-meteo.com/v1/archive"

parametros = {
    "latitude": LATITUD,
    "longitude": LONGITUD,
    "start_date": FECHA_INICIO,
    "end_date": FECHA_FIN,
    "hourly": ",".join(VARIABLES),
    "timezone": "GMT",
    "timeformat": "iso8601",
    "wind_speed_unit": "ms",
    "models": MODELO,
}

print("Solicitando meteorología histórica a Open-Meteo...")
print(f"Periodo: {FECHA_INICIO} a {FECHA_FIN}")
print(f"Modelo solicitado: {MODELO}")

# Ejecuta la petición y muestra un error claro si la API falla.
respuesta = requests.get(url, params=parametros, timeout=120)
respuesta.raise_for_status()
datos = respuesta.json()

# La API puede responder con un objeto de error aunque la petición HTTP haya sido válida.
if datos.get("error"):
    raise RuntimeError(
        f"Open-Meteo devolvió un error: {datos.get('reason', datos)}"
    )

# Conserva la respuesta original para trazabilidad y reproducibilidad.
ruta_json = CARPETA_RAW / "london_historical_weather_era5.json"

with open(ruta_json, "w", encoding="utf-8") as archivo:
    json.dump(datos, archivo, ensure_ascii=False, indent=2)

# Comprueba que la respuesta incluya las observaciones horarias.
horario = datos.get("hourly", {})

if "time" not in horario:
    raise ValueError("La respuesta no contiene la serie horaria esperada.")

# Convierte las series de la API en una tabla con una fila por hora.
df = pd.DataFrame(horario)

# Interpreta las fechas como UTC para poder unirlas con las observaciones de ozono.
df["fecha_utc"] = pd.to_datetime(df.pop("time"), utc=True)

# Ordena por fecha y comprueba que no haya timestamps duplicados.
df = df.sort_values("fecha_utc").reset_index(drop=True)

if df["fecha_utc"].duplicated().any():
    raise ValueError("La respuesta meteorológica contiene fechas duplicadas.")

# Añade metadatos para documentar la fuente y el producto de reanálisis.
df["fuente_meteorologica"] = "Open-Meteo Historical Weather API"
df["modelo_meteorologico"] = MODELO
df["latitud_solicitada"] = LATITUD
df["longitud_solicitada"] = LONGITUD

# Guarda los datos meteorológicos sin rellenar valores ausentes artificialmente.
ruta_csv = CARPETA_TABLAS / "london_historical_weather.csv"
df.to_csv(ruta_csv, index=False)

# Informa la cobertura y las unidades devueltas por la API.
print("\n=== DESCARGA METEOROLÓGICA COMPLETADA ===")
print(f"Filas horarias: {len(df)}")
print(f"Inicio UTC: {df['fecha_utc'].min()}")
print(f"Fin UTC: {df['fecha_utc'].max()}")
print(f"Coordenadas de la rejilla devueltas: "
      f"{datos.get('latitude')}, {datos.get('longitude')}")
print(f"Elevación de la rejilla: {datos.get('elevation')} m")
print(f"Unidades: {datos.get('hourly_units', {})}")

print("\nValores ausentes por variable:")
for columna in VARIABLES:
    if columna in df.columns:
        print(f"{columna}: {df[columna].isna().sum()}")

print(f"\nJSON original: {ruta_json}")
print(f"CSV meteorológico: {ruta_csv}")
