
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = "outputs/tables/o3_pinn_predicciones_detalladas.csv"

SALIDA = "outputs/tables/36_diagnostico_episodios_influyentes.csv"

MODELOS = [
    "MLP_con_meteorologia",
    "Modelo_neuronal_fisico"
]

# Episodios identificados en el diagnóstico anterior.
# Las fechas están expresadas en UTC.
CASOS = [
    (1, "2023-06-30 07:00", "2023-06-30 22:00"),
    (1, "2023-07-01 07:00", "2023-07-08 20:00"),
    (6, "2023-05-22 05:00", "2023-05-23 02:00"),
    (6, "2023-06-25 05:00", "2023-06-29 01:00"),
    (24, "2023-07-02 06:00", "2023-07-08 20:00"),
    (24, "2023-08-02 23:00", "2023-08-07 09:00"),
    (24, "2023-05-05 23:00", "2023-05-06 19:00")
]

# ============================================================
# CARGAR DATOS
# ============================================================

df = pd.read_csv(ARCHIVO)

df["fecha_objetivo"] = pd.to_datetime(
    df["fecha_objetivo"],
    utc=True
)

df = df[
    (df["particion"] == "validation") &
    (df["modelo"].isin(MODELOS))
].copy()

resultados = []

print("\nDIAGNÓSTICO DE EPISODIOS INFLUYENTES")
print("=" * 100)

# ============================================================
# ANALIZAR CADA EPISODIO
# ============================================================

for horizonte, inicio, fin in CASOS:

    inicio = pd.Timestamp(inicio, tz="UTC")
    fin = pd.Timestamp(fin, tz="UTC")

    grupo = df[
        (df["horizonte_horas"] == horizonte) &
        (df["fecha_objetivo"] >= inicio) &
        (df["fecha_objetivo"] <= fin)
    ].copy()

    print("\n" + "-" * 100)
    print(f"HORIZONTE: {horizonte} h")
    print(f"PERIODO: {inicio} a {fin}")
    print("-" * 100)

    if grupo.empty:
        print("No se encontraron observaciones.")
        continue

    # --------------------------------------------------------
    # Verificar fechas comunes entre los dos modelos.
    # --------------------------------------------------------

    fechas_por_modelo = {}

    for modelo in MODELOS:
        fechas_por_modelo[modelo] = set(
            grupo.loc[
                grupo["modelo"] == modelo,
                "fecha_objetivo"
            ]
        )

    fechas_comunes = (
        fechas_por_modelo[MODELOS[0]]
        & fechas_por_modelo[MODELOS[1]]
    )

    grupo = grupo[
        grupo["fecha_objetivo"].isin(fechas_comunes)
    ].copy()

    if not fechas_comunes:
        print("No existen fechas comunes entre los modelos.")
        continue

    # --------------------------------------------------------
    # Calcular métricas individuales.
    # --------------------------------------------------------

    resumen = grupo.groupby("modelo").agg(
        n=("o3_real", "count"),
        o3_real_medio=("o3_real", "mean"),
        o3_predicho_medio=("o3_predicho", "mean"),
        sesgo_medio=("residuo", "mean"),
        mae=("error_absoluto", "mean"),
        mse=("error_cuadratico", "mean")
    )

    resumen["rmse"] = np.sqrt(resumen["mse"])

    resumen = resumen.drop(
        columns=["mse"]
    )

    print("\nMÉTRICAS POR MODELO")
    print(resumen.round(4).to_string())

    # --------------------------------------------------------
    # Comparación directa entre modelos.
    # --------------------------------------------------------

    mlp = resumen.loc[MODELOS[0]]
    fisico = resumen.loc[MODELOS[1]]

    delta_mae = mlp["mae"] - fisico["mae"]
    delta_rmse = mlp["rmse"] - fisico["rmse"]

    print("\nDIFERENCIAS (MLP - FÍSICO)")
    print(f"Delta MAE:  {delta_mae:.4f}")
    print(f"Delta RMSE: {delta_rmse:.4f}")

    # --------------------------------------------------------
    # Dispersión de las concentraciones reales.
    # --------------------------------------------------------

    reales = (
        grupo[["fecha_objetivo", "o3_real"]]
        .drop_duplicates("fecha_objetivo")
    )

    print("\nDISTRIBUCIÓN DE O3 REAL")

    print(
        reales["o3_real"].describe(
            percentiles=[0.25, 0.50, 0.75, 0.90]
        ).round(4).to_string()
    )

    # --------------------------------------------------------
    # Correlación entre predicciones y valores reales.
    # --------------------------------------------------------

    ancho = grupo.pivot(
        index="fecha_objetivo",
        columns="modelo",
        values=["o3_real", "o3_predicho"]
    ).dropna()

    for modelo in MODELOS:

        correlacion = ancho[
            ("o3_real", modelo)
        ].corr(
            ancho[("o3_predicho", modelo)]
        )

        print(
            f"Correlación real-predicho ({modelo}): "
            f"{correlacion:.4f}"
        )

    # --------------------------------------------------------
    # Guardar resultados.
    # --------------------------------------------------------

    for modelo in MODELOS:

        fila = resumen.loc[modelo]

        resultados.append({
            "horizonte": horizonte,
            "fecha_inicio": inicio,
            "fecha_fin": fin,
            "modelo": modelo,
            "n": int(fila["n"]),
            "o3_real_medio": fila["o3_real_medio"],
            "o3_predicho_medio": fila["o3_predicho_medio"],
            "sesgo_medio": fila["sesgo_medio"],
            "mae": fila["mae"],
            "rmse": fila["rmse"],
            "correlacion_real_predicho": ancho[
                ("o3_real", modelo)
            ].corr(
                ancho[("o3_predicho", modelo)]
            )
        })

    print("\nObservaciones comunes:", len(fechas_comunes))

# ============================================================
# EXPORTAR RESULTADOS
# ============================================================

salida = pd.DataFrame(resultados)

Path(SALIDA).parent.mkdir(
    parents=True,
    exist_ok=True
)

salida.to_csv(
    SALIDA,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 100)
print("DIAGNÓSTICO FINALIZADO")
print("Archivo:", SALIDA)
print("Registros guardados:", len(salida))
