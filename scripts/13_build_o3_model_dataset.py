
# Construye una serie horaria regular para el modelado de O3.
# Conserva los valores originales, identifica los huecos y crea una máscara
# para separar las observaciones aptas para entrenamiento.
# Los registros con hasFlags=True se excluyen provisionalmente del objetivo
# de entrenamiento, pero se conservan para una revisión posterior.

import json
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA_ENTRADA = Path("data/raw/openaq_historical_o3")
CARPETA_SALIDA = Path("outputs/tables")
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVO_SALIDA = CARPETA_SALIDA / "o3_dataset_horario.csv"

# --------------------------------------------------
# 1. Cargar todos los registros de los archivos JSON
# --------------------------------------------------

registros = []

archivos = sorted(
    CARPETA_ENTRADA.glob("london_o3_sensor_1304560_*.json")
)

if not archivos:
    raise FileNotFoundError(
        f"No se encontraron archivos en {CARPETA_ENTRADA.resolve()}"
    )

for archivo in archivos:
    with archivo.open("r", encoding="utf-8") as f:
        datos = json.load(f)

    for registro in datos.get("results", []):
        # La fecha se extrae del inicio del periodo de medición.
        periodo = registro.get("period", {})
        fecha_info = (
            periodo.get("datetimeFrom", {})
            if isinstance(periodo, dict) else {}
        )

        fecha = (
            fecha_info.get("utc")
            if isinstance(fecha_info, dict) else None
        )

        # Compatibilidad con formatos alternativos de fecha.
        if not fecha:
            fecha_info = registro.get("datetime", {})
            if isinstance(fecha_info, dict):
                fecha = fecha_info.get("utc")
            elif isinstance(fecha_info, str):
                fecha = fecha_info

        if not fecha:
            continue

        flag_info = registro.get("flagInfo", {})
        has_flags = (
            flag_info.get("hasFlags")
            if isinstance(flag_info, dict) else None
        )

        parametro = registro.get("parameter", {})
        unidad = (
            parametro.get("units")
            if isinstance(parametro, dict) else None
        )

        registros.append({
            "fecha_utc": pd.to_datetime(
                fecha, utc=True, errors="coerce"
            ),
            "valor_o3_original": pd.to_numeric(
                registro.get("value"), errors="coerce"
            ),
            "hasFlags": has_flags,
            "unidad": unidad,
            "archivo_origen": archivo.name
        })

df = pd.DataFrame(registros)
df = df.dropna(subset=["fecha_utc"])
df = df.sort_values("fecha_utc")

if df.empty:
    raise ValueError("No se encontraron observaciones con fechas válidas.")

# --------------------------------------------------
# 2. Verificar marcas de tiempo duplicadas
# --------------------------------------------------

duplicados = df["fecha_utc"].duplicated(keep=False)

if duplicados.any():
    print("Se encontraron marcas de tiempo duplicadas:")
    print(
        df.loc[duplicados, ["fecha_utc", "valor_o3_original", "archivo_origen"]]
        .to_string(index=False)
    )
    raise ValueError(
        "Resolver los duplicados antes de construir la serie horaria."
    )

# --------------------------------------------------
# 3. Crear una cuadrícula horaria continua
# --------------------------------------------------

inicio = df["fecha_utc"].min().floor("h")
fin = df["fecha_utc"].max().floor("h")

horas = pd.date_range(
    start=inicio,
    end=fin,
    freq="h",
    tz="UTC"
)

# Usar la fecha como índice permite identificar las horas ausentes.
df = df.set_index("fecha_utc")
df = df.reindex(horas)
df.index.name = "fecha_utc"

# Una fila con observación API indica que la hora aparece en los registros.
df["observacion_api"] = df["archivo_origen"].notna()

# --------------------------------------------------
# 4. Crear máscaras de calidad
# --------------------------------------------------

# La observación tiene un valor numérico disponible.
df["valor_numerico_disponible"] = df["valor_o3_original"].notna()

# True solo cuando OpenAQ indica explícitamente que hay banderas.
df["bandera_activa"] = df["hasFlags"].eq(True)

# Se requiere hasFlags=False para considerar el registro apto inicialmente.
# Los valores marcados quedan preservados, pero fuera del objetivo provisional.
df["observacion_entrenable"] = (
    df["valor_numerico_disponible"]
    & df["hasFlags"].eq(False)
)

# Crear la variable objetivo sin sobrescribir el dato original.
df["valor_o3_modelo"] = df["valor_o3_original"].where(
    df["observacion_entrenable"],
    np.nan
)

# --------------------------------------------------
# 5. Crear variables temporales básicas
# --------------------------------------------------

# Estas variables ayudan a representar la periodicidad diaria y anual.
hora_decimal = (
    df.index.hour
    + df.index.minute / 60.0
)

# Extrae el día del año usando el nombre de propiedad actualizado de Pandas.
dia_del_anio = df.index.day_of_year

df["hora_sin"] = np.sin(2 * np.pi * hora_decimal / 24)
df["hora_cos"] = np.cos(2 * np.pi * hora_decimal / 24)

df["dia_anio_sin"] = np.sin(2 * np.pi * dia_del_anio / 365.25)
df["dia_anio_cos"] = np.cos(2 * np.pi * dia_del_anio / 365.25)

# --------------------------------------------------
# 6. Guardar la serie horaria
# --------------------------------------------------

columnas_salida = [
    "valor_o3_original",
    "valor_o3_modelo",
    "hasFlags",
    "bandera_activa",
    "observacion_api",
    "valor_numerico_disponible",
    "observacion_entrenable",
    "unidad",
    "archivo_origen",
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos"
]

df[columnas_salida].to_csv(
    ARCHIVO_SALIDA,
    index=True,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 7. Imprimir el resumen de control
# --------------------------------------------------

total_horas = len(df)
horas_con_registro = int(df["observacion_api"].sum())
horas_con_valor = int(df["valor_numerico_disponible"].sum())
horas_entrenables = int(df["observacion_entrenable"].sum())
horas_sin_registro = int((~df["observacion_api"]).sum())
horas_con_bandera = int(df["bandera_activa"].sum())

print("\n=== CONJUNTO HORARIO DE O3 ===")
print("Sensor: 1304560")
print("Inicio UTC:", inicio)
print("Fin UTC:", fin)
print("Horas en la cuadrícula:", total_horas)
print("Horas con registro de API:", horas_con_registro)
print("Horas con valor numérico:", horas_con_valor)
print("Horas con bandera activa:", horas_con_bandera)
print("Horas aptas provisionalmente para entrenamiento:", horas_entrenables)
print("Horas sin registro de API:", horas_sin_registro)
print(
    "Porcentaje de horas entrenables:",
    round(100 * horas_entrenables / total_horas, 2), "%"
)

print("\nArchivo generado:")
print(ARCHIVO_SALIDA)
