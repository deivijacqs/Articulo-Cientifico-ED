
# Inspecciona las mediciones cercanas a los cinco valores máximos de O3.
# Muestra fecha, concentración, bandera y distancia temporal entre registros.
# No elimina valores ni modifica los archivos originales.

import json
from pathlib import Path

import pandas as pd

CARPETA = Path("data/raw/openaq_historical_o3")

# --------------------------------------------------
# 1. Cargar todas las observaciones
# --------------------------------------------------

registros = []

for archivo in sorted(CARPETA.glob("london_o3_sensor_1304560_*.json")):
    with archivo.open("r", encoding="utf-8") as f:
        datos = json.load(f)

    for registro in datos.get("results", []):
        # Obtener la fecha UTC del inicio del periodo de medición.
        periodo = registro.get("period", {})
        fecha_info = (
            periodo.get("datetimeFrom", {})
            if isinstance(periodo, dict) else {}
        )

        fecha = (
            fecha_info.get("utc")
            if isinstance(fecha_info, dict) else None
        )

        # Compatibilidad con estructuras alternativas.
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

        registros.append({
            "fecha_utc": pd.to_datetime(fecha, utc=True, errors="coerce"),
            "valor_o3": pd.to_numeric(
                registro.get("value"), errors="coerce"
            ),
            "hasFlags": has_flags,
            "archivo": archivo.name
        })

df = pd.DataFrame(registros)
df = df.dropna(subset=["fecha_utc"])
df = df.sort_values("fecha_utc").reset_index(drop=True)

# --------------------------------------------------
# 2. Seleccionar los cinco valores máximos
# --------------------------------------------------

picos = (
    df.dropna(subset=["valor_o3"])
    .nlargest(5, "valor_o3")
    .sort_values("valor_o3", ascending=False)
)

print("\n=== CINCO VALORES MÁXIMOS ===")
print(
    picos[["fecha_utc", "valor_o3", "hasFlags"]]
    .to_string(index=False)
)

# --------------------------------------------------
# 3. Mostrar las mediciones cercanas a cada pico
# --------------------------------------------------

for numero, (_, pico) in enumerate(picos.iterrows(), start=1):
    fecha_pico = pico["fecha_utc"]
    valor_pico = pico["valor_o3"]

    # Examinar doce horas antes y doce horas después.
    inicio = fecha_pico - pd.Timedelta(hours=12)
    fin = fecha_pico + pd.Timedelta(hours=12)

    ventana = df[
        df["fecha_utc"].between(inicio, fin)
    ].copy()

    # Calcular el tiempo desde la observación anterior.
    ventana["intervalo_horas"] = (
        ventana["fecha_utc"].diff().dt.total_seconds() / 3600
    )

    print("\n" + "=" * 75)
    print(f"PICO {numero}: {valor_pico} µg/m³ en {fecha_pico}")
    print("=" * 75)

    print(
        ventana[
            ["fecha_utc", "valor_o3", "hasFlags", "intervalo_horas"]
        ].to_string(index=False)
    )

    valores_ventana = ventana["valor_o3"].dropna()

    if not valores_ventana.empty:
        print("\nEstadísticas de la ventana:")
        print(f"Mediciones numéricas: {len(valores_ventana)}")
        print(f"Mínimo: {valores_ventana.min():.2f} µg/m³")
        print(f"Mediana: {valores_ventana.median():.2f} µg/m³")
        print(f"Máximo: {valores_ventana.max():.2f} µg/m³")

print("\nInspección terminada. No se modificaron los datos.")
