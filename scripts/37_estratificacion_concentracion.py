
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = "outputs/tables/o3_pinn_predicciones_detalladas.csv"

SALIDA = "outputs/tables/37_estratificacion_concentracion.csv"

MODELO_MLP = "MLP_con_meteorologia"
MODELO_FISICO = "Modelo_neuronal_fisico"

MODELOS = [MODELO_MLP, MODELO_FISICO]

HORIZONTES = [1, 6, 24]

# ============================================================
# CARGAR DATOS
# ============================================================

df = pd.read_csv(ARCHIVO)

df["fecha_objetivo"] = pd.to_datetime(
    df["fecha_objetivo"],
    utc=True
)

# Trabajar exclusivamente con validación.
df = df[
    (df["particion"] == "validation") &
    (df["modelo"].isin(MODELOS)) &
    (df["horizonte_horas"].isin(HORIZONTES))
].copy()

resultados = []

print("\nESTRATIFICACIÓN POR CONCENTRACIÓN DE O3")
print("=" * 100)

# ============================================================
# ANALIZAR CADA HORIZONTE
# ============================================================

for horizonte in HORIZONTES:

    datos = df[
        df["horizonte_horas"] == horizonte
    ].copy()

    # --------------------------------------------------------
    # Crear una tabla independiente para cada modelo.
    # Así evitamos trabajar con columnas MultiIndex.
    # --------------------------------------------------------

    mlp = (
        datos[datos["modelo"] == MODELO_MLP]
        .set_index("fecha_objetivo")
        [["o3_real", "o3_predicho"]]
        .rename(columns={
            "o3_real": "o3_real_mlp",
            "o3_predicho": "pred_mlp"
        })
    )

    fisico = (
        datos[datos["modelo"] == MODELO_FISICO]
        .set_index("fecha_objetivo")
        [["o3_real", "o3_predicho"]]
        .rename(columns={
            "o3_real": "o3_real_fisico",
            "o3_predicho": "pred_fisico"
        })
    )

    # --------------------------------------------------------
    # Emparejar exactamente las fechas de ambos modelos.
    # --------------------------------------------------------

    tabla = mlp.join(
        fisico,
        how="inner"
    )

    # Eliminar filas incompletas.
    tabla = tabla.dropna().copy()

    if tabla.empty:
        print(f"\nNo hay observaciones comunes para horizonte {horizonte} h.")
        continue

    # --------------------------------------------------------
    # Verificar que ambos modelos tienen el mismo O3 real.
    # --------------------------------------------------------

    diferencia_reales = (
        tabla["o3_real_mlp"] -
        tabla["o3_real_fisico"]
    ).abs()

    if (diferencia_reales > 1e-8).any():
        raise ValueError(
            f"Los valores reales no coinciden en horizonte {horizonte}."
        )

    # Utilizar una única concentración real.
    tabla["o3_real"] = tabla["o3_real_mlp"]

    # --------------------------------------------------------
    # Calcular los umbrales de concentración.
    # --------------------------------------------------------

    p33 = float(
        tabla["o3_real"].quantile(1 / 3)
    )

    p67 = float(
        tabla["o3_real"].quantile(2 / 3)
    )

    print("\n" + "=" * 100)
    print(f"HORIZONTE: {horizonte} h")
    print("=" * 100)

    print(f"Observaciones comunes: {len(tabla)}")
    print(f"Percentil 33: {p33:.4f} µg/m³")
    print(f"Percentil 67: {p67:.4f} µg/m³")

    # --------------------------------------------------------
    # Clasificar las observaciones según el O3 real.
    # Los mismos umbrales se aplican a ambos modelos.
    # --------------------------------------------------------

    tabla["estrato"] = np.select(
        [
            tabla["o3_real"] <= p33,
            tabla["o3_real"] <= p67
        ],
        [
            "Baja",
            "Media"
        ],
        default="Alta"
    )

    # --------------------------------------------------------
    # Analizar cada estrato.
    # --------------------------------------------------------

    for estrato in ["Baja", "Media", "Alta"]:

        grupo = tabla[
            tabla["estrato"] == estrato
        ].copy()

        if grupo.empty:
            continue

        reales = grupo["o3_real"]

        print("\n" + "-" * 90)
        print(f"ESTRATO: {estrato}")
        print(f"Observaciones: {len(grupo)}")
        print("-" * 90)

        metricas = {}

        # ----------------------------------------------------
        # Calcular métricas de cada modelo.
        # ----------------------------------------------------

        for modelo in MODELOS:

            if modelo == MODELO_MLP:
                predicciones = grupo["pred_mlp"]
            else:
                predicciones = grupo["pred_fisico"]

            # Residuo = real - predicho.
            residuos = reales - predicciones

            mae = float(residuos.abs().mean())

            rmse = float(
                np.sqrt((residuos ** 2).mean())
            )

            sesgo = float(residuos.mean())

            correlacion = float(
                reales.corr(predicciones)
            )

            metricas[modelo] = {
                "n": len(grupo),
                "o3_real_medio": float(reales.mean()),
                "o3_predicho_medio": float(predicciones.mean()),
                "sesgo_medio": sesgo,
                "mae": mae,
                "rmse": rmse,
                "correlacion": correlacion
            }

            print(f"\n{modelo}")
            print(
                f"  O3 real medio:       "
                f"{reales.mean():.4f}"
            )
            print(
                f"  O3 predicho medio:   "
                f"{predicciones.mean():.4f}"
            )
            print(f"  Sesgo medio:         {sesgo:.4f}")
            print(f"  MAE:                 {mae:.4f}")
            print(f"  RMSE:                {rmse:.4f}")
            print(f"  Correlación:         {correlacion:.4f}")

        # ----------------------------------------------------
        # Diferencias entre modelos.
        # Positivo favorece al modelo físico.
        # ----------------------------------------------------

        delta_mae = (
            metricas[MODELO_MLP]["mae"] -
            metricas[MODELO_FISICO]["mae"]
        )

        delta_rmse = (
            metricas[MODELO_MLP]["rmse"] -
            metricas[MODELO_FISICO]["rmse"]
        )

        delta_sesgo = (
            metricas[MODELO_MLP]["sesgo_medio"] -
            metricas[MODELO_FISICO]["sesgo_medio"]
        )

        print("\nCOMPARACIÓN (MLP - FÍSICO)")
        print(f"  Delta MAE:   {delta_mae:.4f}")
        print(f"  Delta RMSE:  {delta_rmse:.4f}")
        print(f"  Delta sesgo: {delta_sesgo:.4f}")

        # ----------------------------------------------------
        # Guardar resultados del estrato.
        # ----------------------------------------------------

        resultados.append({
            "horizonte": horizonte,
            "estrato": estrato,
            "n": len(grupo),
            "umbral_p33": p33,
            "umbral_p67": p67,
            "o3_real_medio": float(reales.mean()),

            "sesgo_mlp": metricas[MODELO_MLP]["sesgo_medio"],
            "sesgo_fisico": metricas[MODELO_FISICO]["sesgo_medio"],

            "mae_mlp": metricas[MODELO_MLP]["mae"],
            "mae_fisico": metricas[MODELO_FISICO]["mae"],

            "rmse_mlp": metricas[MODELO_MLP]["rmse"],
            "rmse_fisico": metricas[MODELO_FISICO]["rmse"],

            "correlacion_mlp": metricas[MODELO_MLP]["correlacion"],
            "correlacion_fisico": metricas[MODELO_FISICO]["correlacion"],

            "delta_mae_mlp_menos_fisico": delta_mae,
            "delta_rmse_mlp_menos_fisico": delta_rmse,
            "delta_sesgo_mlp_menos_fisico": delta_sesgo
        })

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
print("ANÁLISIS FINALIZADO")
print("=" * 100)

print("Archivo generado:", SALIDA)
print("Total de estratos analizados:", len(salida))
