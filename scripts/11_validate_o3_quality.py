
# Valida las banderas reales de calidad de OpenAQ.
# Examina los valores extremos y genera un resumen mensual corregido.
# No elimina ni modifica los archivos originales.

import json
from pathlib import Path

import pandas as pd

CARPETA = Path("data/raw/openaq_historical_o3")
SALIDA = Path("outputs/tables")
SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVOS = sorted(CARPETA.glob("london_o3_sensor_1304560_*.json"))

registros = []

# --------------------------------------------------
# 1. Leer los registros y extraer las banderas reales
# --------------------------------------------------

for archivo in ARCHIVOS:
    with archivo.open("r", encoding="utf-8") as f:
        datos = json.load(f)

    for registro in datos.get("results", []):
        periodo = registro.get("period", {})
        fecha_info = (
            periodo.get("datetimeFrom", {})
            if isinstance(periodo, dict) else {}
        )

        fecha = (
            fecha_info.get("utc")
            if isinstance(fecha_info, dict) else None
        )

        # Alternativa para posibles formatos distintos.
        if not fecha:
            fecha_info = registro.get("datetime", {})
            if isinstance(fecha_info, dict):
                fecha = fecha_info.get("utc")
            elif isinstance(fecha_info, str):
                fecha = fecha_info

        if not fecha:
            continue

        # Extraer hasFlags sin confundirlo con un diccionario no vacío.
        flag_info = registro.get("flagInfo")
        if isinstance(flag_info, dict):
            has_flags = flag_info.get("hasFlags")
        else:
            has_flags = None

        # Conservar el contenido completo para inspeccionar las banderas.
        valor = pd.to_numeric(registro.get("value"), errors="coerce")

        parametro = registro.get("parameter", {})
        unidad = (
            parametro.get("units")
            if isinstance(parametro, dict) else None
        )

        registros.append({
            "fecha_utc": pd.to_datetime(fecha, utc=True, errors="coerce"),
            "valor_o3": valor,
            "hasFlags": has_flags,
            "flagInfo": flag_info,
            "unidad": unidad,
            "archivo": archivo.name
        })

df = pd.DataFrame(registros)
df = df.dropna(subset=["fecha_utc"]).sort_values("fecha_utc")
df["mes"] = df["fecha_utc"].dt.strftime("%Y-%m")

# --------------------------------------------------
# 2. Resumen correcto de banderas y valores nulos
# --------------------------------------------------

# Solo True significa que el registro declara tener banderas.
marcados = df["hasFlags"].eq(True)
valores_nulos = df["valor_o3"].isna()

print("\n=== RESUMEN GLOBAL ===")
print("Registros con fecha válida:", len(df))
print("Valores numéricos:", int(df["valor_o3"].notna().sum()))
print("Valores nulos:", int(valores_nulos.sum()))
print("Registros con hasFlags=True:", int(marcados.sum()))
print("Registros con hasFlags=False:", int(df["hasFlags"].eq(False).sum()))
print("Registros sin hasFlags interpretable:", int(df["hasFlags"].isna().sum()))

print("\n=== RELACIÓN ENTRE BANDERAS Y VALORES NULOS ===")
print(pd.crosstab(
    marcados.rename("tiene_banderas"),
    valores_nulos.rename("valor_nulo"),
    margins=True
))

# Mostrar los contenidos reales de las banderas activas.
print("\n=== EJEMPLOS CON hasFlags=True ===")
ejemplos = df.loc[marcados, [
    "fecha_utc", "valor_o3", "flagInfo", "unidad", "archivo"
]].head(15)

if ejemplos.empty:
    print("No se encontraron registros con hasFlags=True.")
else:
    print(ejemplos.to_string(index=False))

# --------------------------------------------------
# 3. Inspeccionar los valores más altos
# --------------------------------------------------

print("\n=== 15 VALORES MÁS ALTOS ===")

extremos = df.loc[df["valor_o3"].notna()].nlargest(
    15, "valor_o3"
)[[
    "fecha_utc", "valor_o3", "hasFlags", "flagInfo", "unidad", "archivo"
]]

print(extremos.to_string(index=False))

# Guardar los extremos para revisión posterior.
extremos.to_csv(
    SALIDA / "o3_valores_extremos_para_revision.csv",
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 4. Resumen mensual corregido
# --------------------------------------------------

resumen = df.groupby("mes").agg(
    registros=("fecha_utc", "size"),
    valores_nulos=("valor_o3", lambda s: int(s.isna().sum())),
    valores_validos=("valor_o3", lambda s: int(s.notna().sum())),
    banderas_activas=("hasFlags", lambda s: int(s.eq(True).sum())),
    fechas_unicas=("fecha_utc", "nunique"),
    minimo=("valor_o3", "min"),
    mediana=("valor_o3", "median"),
    maximo=("valor_o3", "max")
).reset_index()

resumen.to_csv(
    SALIDA / "o3_auditoria_mensual_corregida.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== RESUMEN MENSUAL CORREGIDO ===")
print(resumen.to_string(index=False))

print("\nArchivo guardado:")
print(SALIDA / "o3_auditoria_mensual_corregida.csv")
print(SALIDA / "o3_valores_extremos_para_revision.csv")
