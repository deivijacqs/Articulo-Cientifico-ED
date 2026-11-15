
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = "outputs/tables/o3_pinn_predicciones_detalladas.csv"

SALIDA = "outputs/tables/38_estratificacion_cinco_cuantiles.csv"

MODELO_MLP = "MLP_con_meteorologia"
MODELO_FISICO = "Modelo_neuronal_fisico"

MODELOS = [MODELO_MLP, MODELO_FISICO]

HORIZONTES = [1, 6, 24]

CUANTILES = [0.20, 0.40, 0.60, 0.80]

NOMBRES_ESTRATOS = ["Q1", "Q2", "Q3", "Q4", "Q5"]

# ============================================================
# CARGAR DATOS
# ============================================================

df = pd.read_csv(ARCHIVO)

df["fecha_objetivo"] = pd.to_datetime(
    df["fecha_objetivo"],
    utc=True
)

# Utilizar únicamente validación y los dos modelos.
df = df[
    (df["particion"] == "validation") &
    (df["modelo"].isin(MODELOS)) &
    (df["horizonte_horas"].isin(HORIZONTES))
].copy()

resultados = []
umbrales_resultados = []

print("\nESTRATIFICACIÓN POR CINCO CUANTILES DE O3")
print("=" * 110)

# ============================================================
# ANALIZAR CADA HORIZONTE
# ============================================================

for horizonte in HORIZONTES:

    datos = df[
        df["horizonte_horas"] == horizonte
    ].copy()

    # --------------------------------------------------------
    # Separar las predicciones de cada modelo.
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

    # Emparejar exactamente las fechas.
    tabla = mlp.join(
        fisico,
        how="inner"
    ).dropna().copy()

    if tabla.empty:
        print(f"\nSin datos comunes para horizonte {horizonte} h.")
        continue

    # --------------------------------------------------------
    # Verificar que ambos modelos utilizan el mismo O3 real.
    # --------------------------------------------------------

    diferencia_reales = (
        tabla["o3_real_mlp"] -
        tabla["o3_real_fisico"]
    ).abs()

    if (diferencia_reales > 1e-8).any():
        raise ValueError(
            f"Los valores reales no coinciden en horizonte {horizonte}."
        )

    tabla["o3_real"] = tabla["o3_real_mlp"]

    # --------------------------------------------------------
    # Calcular los puntos de corte de los cinco cuantiles.
    # --------------------------------------------------------

    umbrales = [
        float(tabla["o3_real"].quantile(q))
        for q in CUANTILES
    ]

    print("\n" + "=" * 110)
    print(f"HORIZONTE: {horizonte} h")
    print("=" * 110)

    print(f"Observaciones comunes: {len(tabla)}")

    print("\nPUNTOS DE CORTE")

    for q, valor in zip(CUANTILES, umbrales):
        print(f"Percentil {q * 100:.0f}: {valor:.4f} µg/m³")

    # --------------------------------------------------------
    # Clasificar las observaciones.
    #
    # Se mantienen juntas las concentraciones idénticas.
    # Si varios cuantiles tienen el mismo punto de corte,
    # algún estrato podría quedar vacío.
    # --------------------------------------------------------

    condiciones = [
        tabla["o3_real"] <= umbrales[0],
        (tabla["o3_real"] > umbrales[0]) &
        (tabla["o3_real"] <= umbrales[1]),
        (tabla["o3_real"] > umbrales[1]) &
        (tabla["o3_real"] <= umbrales[2]),
        (tabla["o3_real"] > umbrales[2]) &
        (tabla["o3_real"] <= umbrales[3])
    ]

    tabla["estrato"] = np.select(
        condiciones,
        NOMBRES_ESTRATOS[:4],
        default="Q5"
    )

    # Guardar los umbrales para documentar la estratificación.
    umbrales_resultados.append({
        "horizonte": horizonte,
        "observaciones_comunes": len(tabla),
        "p20": umbrales[0],
        "p40": umbrales[1],
        "p60": umbrales[2],
        "p80": umbrales[3]
    })

    # --------------------------------------------------------
    # Evaluar cada cuantíl.
    # --------------------------------------------------------

    for i, estrato in enumerate(NOMBRES_ESTRATOS):

        grupo = tabla[
            tabla["estrato"] == estrato
        ].copy()

        print("\n" + "-" * 100)
        print(f"ESTRATO: {estrato}")
        print(f"Observaciones: {len(grupo)}")
        print("-" * 100)

        if grupo.empty:
            print("Estrato vacío debido a valores repetidos en los cortes.")
            continue

        reales = grupo["o3_real"]

        # Límites de concentración del estrato.
        minimo = float(reales.min())
        maximo = float(reales.max())
        media_real = float(reales.mean())

        print(
            f"Intervalo observado: {minimo:.4f} - "
            f"{maximo:.4f} µg/m³"
        )

        print(f"O3 real medio: {media_real:.4f} µg/m³")

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

            correlacion = reales.corr(predicciones)

            if pd.isna(correlacion):
                correlacion = np.nan
            else:
                correlacion = float(correlacion)

            metricas[modelo] = {
                "mae": mae,
                "rmse": rmse,
                "sesgo": sesgo,
                "correlacion": correlacion,
                "predicho_medio": float(predicciones.mean())
            }

            print(f"\n{modelo}")
            print(
                f"  Predicción media: {metricas[modelo]['predicho_medio']:.4f}"
            )
            print(f"  Sesgo:            {sesgo:.4f}")
            print(f"  MAE:              {mae:.4f}")
            print(f"  RMSE:             {rmse:.4f}")
            print(f"  Correlación:      {correlacion:.4f}")

        # ----------------------------------------------------
        # Comparación directa.
        # Positivo favorece al modelo físico.
        # ----------------------------------------------------

        m_mlp = metricas[MODELO_MLP]
        m_fisico = metricas[MODELO_FISICO]

        delta_mae = m_mlp["mae"] - m_fisico["mae"]
        delta_rmse = m_mlp["rmse"] - m_fisico["rmse"]

        delta_sesgo = m_mlp["sesgo"] - m_fisico["sesgo"]

        # Mejora porcentual respecto al error del MLP.
        mejora_mae_pct = (
            100 * delta_mae / m_mlp["mae"]
            if m_mlp["mae"] != 0 else np.nan
        )

        mejora_rmse_pct = (
            100 * delta_rmse / m_mlp["rmse"]
            if m_mlp["rmse"] != 0 else np.nan
        )

        print("\nCOMPARACIÓN (MLP - FÍSICO)")
        print(f"  Delta MAE:          {delta_mae:.4f}")
        print(f"  Delta RMSE:         {delta_rmse:.4f}")
        print(f"  Delta sesgo:        {delta_sesgo:.4f}")
        print(f"  Mejora MAE:         {mejora_mae_pct:.2f}%")
        print(f"  Mejora RMSE:        {mejora_rmse_pct:.2f}%")

        # ----------------------------------------------------
        # Guardar resultados.
        # ----------------------------------------------------

        resultados.append({
            "horizonte": horizonte,
            "estrato": estrato,
            "n": len(grupo),
            "concentracion_min": minimo,
            "concentracion_max": maximo,
            "o3_real_medio": media_real,

            "sesgo_mlp": m_mlp["sesgo"],
            "sesgo_fisico": m_fisico["sesgo"],

            "mae_mlp": m_mlp["mae"],
            "mae_fisico": m_fisico["mae"],

            "rmse_mlp": m_mlp["rmse"],
            "rmse_fisico": m_fisico["rmse"],

            "correlacion_mlp": m_mlp["correlacion"],
            "correlacion_fisico": m_fisico["correlacion"],

            "delta_mae": delta_mae,
            "delta_rmse": delta_rmse,
            "delta_sesgo": delta_sesgo,

            "mejora_mae_pct": mejora_mae_pct,
            "mejora_rmse_pct": mejora_rmse_pct
        })

# ============================================================
# EXPORTAR RESULTADOS
# ============================================================

Path(SALIDA).parent.mkdir(
    parents=True,
    exist_ok=True
)

salida = pd.DataFrame(resultados)

salida.to_csv(
    SALIDA,
    index=False,
    encoding="utf-8"
)

# Guardar también los puntos de corte.
salida_umbrales = pd.DataFrame(umbrales_resultados)

archivo_umbrales = (
    "outputs/tables/38_umbrales_cuantiles.csv"
)

salida_umbrales.to_csv(
    archivo_umbrales,
    index=False,
    encoding="utf-8"
)

# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "=" * 110)
print("ANÁLISIS FINALIZADO")
print("=" * 110)

print("Resultados:", SALIDA)
print("Umbrales:", archivo_umbrales)
print("Estratos generados:", len(salida))

print("\nRESUMEN DE DIFERENCIAS DE MAE")

if not salida.empty:
    print(
        salida.pivot(
            index="estrato",
            columns="horizonte",
            values="delta_mae"
        ).round(4).to_string()
    )
