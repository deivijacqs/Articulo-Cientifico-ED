
# Inspecciona la estructura real de las banderas de calidad y los intervalos.
# No modifica los JSON ni elimina observaciones.

import json
from pathlib import Path
from collections import Counter

import pandas as pd

CARPETA = Path("data/raw/openaq_historical_o3")
ARCHIVOS = sorted(CARPETA.glob("london_o3_sensor_1304560_*.json"))

registros = []
tipos_banderas = Counter()
ejemplos_banderas = []
claves_registros = Counter()

# --------------------------------------------------
# 1. Leer los registros originales
# --------------------------------------------------

for archivo in ARCHIVOS:
    with archivo.open("r", encoding="utf-8") as f:
        datos = json.load(f)

    for registro in datos.get("results", []):
        claves_registros.update(registro.keys())

        # Guardar ejemplos de cómo OpenAQ representa las banderas.
        bandera = registro.get("flagInfo")
        tipo = type(bandera).__name__
        tipos_banderas[tipo] += 1

        if bandera not in (None, {}, []):
            if len(ejemplos_banderas) < 8:
                ejemplos_banderas.append({
                    "archivo": archivo.name,
                    "flagInfo": bandera,
                    "valor": registro.get("value")
                })

        # Extraer la fecha según la estructura del registro.
        periodo = registro.get("period", {})
        fecha = None

        if isinstance(periodo, dict):
            fecha_info = periodo.get("datetimeFrom", {})
            if isinstance(fecha_info, dict):
                fecha = fecha_info.get("utc")

        if fecha is None:
            fecha_info = registro.get("datetime", {})
            if isinstance(fecha_info, dict):
                fecha = fecha_info.get("utc")
            elif isinstance(fecha_info, str):
                fecha = fecha_info

        if fecha:
            registros.append({
                "fecha_utc": pd.to_datetime(
                    fecha, utc=True, errors="coerce"
                ),
                "valor": pd.to_numeric(
                    registro.get("value"), errors="coerce"
                ),
                "flagInfo": bandera,
                "archivo": archivo.name
            })

# --------------------------------------------------
# 2. Mostrar estructura de las banderas
# --------------------------------------------------

print("\n=== CLAVES PRESENTES EN LOS REGISTROS ===")
print(claves_registros)

print("\n=== TIPOS DE flagInfo ===")
print(tipos_banderas)

print("\n=== EJEMPLOS DE REGISTROS CON flagInfo ===")
for ejemplo in ejemplos_banderas:
    print(ejemplo)

# --------------------------------------------------
# 3. Revisar duplicados entre todos los archivos
# --------------------------------------------------

df = pd.DataFrame(registros)
df = df.dropna(subset=["fecha_utc"]).sort_values("fecha_utc")

print("\n=== CONTROL GLOBAL DE FECHAS ===")
print("Registros con fecha válida:", len(df))
print("Marcas de tiempo únicas:", df["fecha_utc"].nunique())
print("Registros en marcas de tiempo duplicadas:",
      int(df["fecha_utc"].duplicated(keep=False).sum()))

# --------------------------------------------------
# 4. Revisar intervalos entre mediciones
# --------------------------------------------------

fechas_unicas = (
    df["fecha_utc"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

intervalos = fechas_unicas.diff().dropna()

print("\n=== INTERVALOS ENTRE MARCAS DE TIEMPO ===")
print("\nIntervalos más frecuentes:")
print(intervalos.value_counts().head(10).to_string())

huecos = pd.DataFrame({
    "fecha_anterior": fechas_unicas.shift(1),
    "fecha_siguiente": fechas_unicas,
    "intervalo": intervalos
}).dropna()

huecos_largos = huecos[
    huecos["intervalo"] > pd.Timedelta(hours=2)
].copy()

huecos_largos["horas_sin_registro"] = (
    huecos_largos["intervalo"].dt.total_seconds() / 3600
)

print("\nCantidad de intervalos superiores a 2 horas:",
      len(huecos_largos))

print("\nLos 20 huecos más largos:")
print(
    huecos_largos.nlargest(20, "horas_sin_registro")[
        ["fecha_anterior", "fecha_siguiente", "horas_sin_registro"]
    ].to_string(index=False)
)

# --------------------------------------------------
# 5. Examinar valores extremos, sin eliminarlos
# --------------------------------------------------

valores = df["valor"].dropna()

print("\n=== VALORES NUMÉRICOS ===")
print("Valores válidos:", len(valores))
print("Valores nulos:", int(df["valor"].isna().sum()))

if not valores.empty:
    print("Mínimo:", valores.min())
    print("Mediana:", valores.median())
    print("Máximo:", valores.max())
    print("\nCuantiles:")
    print(valores.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))

print("\nInspección terminada. No se modificaron los datos.")
