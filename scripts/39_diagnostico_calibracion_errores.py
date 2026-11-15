
# ============================================================
# SCRIPT 39
# DIAGNÓSTICO DE CALIBRACIÓN Y DISTRIBUCIÓN DE ERRORES
#
# Objetivo:
# Determinar si las diferencias entre el MLP con meteorología
# y el modelo neuronal físico se relacionan principalmente
# con el sesgo sistemático o con la dispersión de los errores.
#
# Se trabaja exclusivamente con VALIDACIÓN.
# No se modifican ni reentrenan los modelos.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. CONFIGURACIÓN
# ------------------------------------------------------------

ARCHIVO_ENTRADA = Path(
    "outputs/tables/o3_pinn_predicciones_detalladas.csv"
)

DIRECTORIO_SALIDA = Path("outputs/tables")

ARCHIVO_SALIDA = (
    DIRECTORIO_SALIDA /
    "39_diagnostico_calibracion_errores.csv"
)

PARTICION = "validation"

MODELOS = [
    "MLP_con_meteorologia",
    "Modelo_neuronal_fisico"
]

HORIZONTES = [1, 6, 24]

VARIABLE_REAL = "o3_real"
VARIABLE_PREDICHA = "o3_predicho"

SEMILLA = 42


# ------------------------------------------------------------
# 2. FUNCIONES AUXILIARES
# ------------------------------------------------------------

def calcular_metricas(grupo):
    """
    Calcula métricas de error y calibración para un modelo
    dentro de un horizonte y estrato de concentración.
    """

    real = grupo[VARIABLE_REAL].to_numpy(dtype=float)
    predicho = grupo[VARIABLE_PREDICHA].to_numpy(dtype=float)

    # Error positivo: subestimación.
    # Error negativo: sobreestimación.
    error = real - predicho

    error_absoluto = np.abs(error)

    sesgo = np.mean(error)
    varianza_error = np.var(error, ddof=0)

    mse = np.mean(error ** 2)

    # Descomposición exacta del MSE:
    # MSE = sesgo^2 + varianza del error.
    contribucion_sesgo = sesgo ** 2

    # Proporción del MSE explicada por el componente sistemático.
    proporcion_sesgo = (
        contribucion_sesgo / mse * 100
        if mse > 0 else np.nan
    )

    # Regresión de calibración:
    # valor observado = intercepto + pendiente * predicción.
    #
    # Idealmente:
    # intercepto = 0
    # pendiente = 1

    if (
        len(real) >= 3
        and np.std(predicho) > 0
        and np.std(real) > 0
    ):
        pendiente, intercepto = np.polyfit(
            predicho,
            real,
            1
        )
    else:
        pendiente = np.nan
        intercepto = np.nan

    # Correlación entre valores reales y predichos.
    correlacion = (
        np.corrcoef(real, predicho)[0, 1]
        if np.std(real) > 0 and np.std(predicho) > 0
        else np.nan
    )

    return {
        "observaciones": len(grupo),

        "o3_real_medio": np.mean(real),
        "o3_predicho_medio": np.mean(predicho),

        "sesgo": sesgo,
        "sesgo_absoluto": abs(sesgo),

        "mae": np.mean(error_absoluto),
        "rmse": np.sqrt(mse),

        "mediana_error_absoluto": np.median(error_absoluto),
        "p90_error_absoluto": np.percentile(error_absoluto, 90),

        "desviacion_error": np.std(error, ddof=0),
        "varianza_error": varianza_error,

        "mse": mse,
        "contribucion_sesgo": contribucion_sesgo,
        "proporcion_mse_sesgo_pct": proporcion_sesgo,

        "porcentaje_subestimacion": np.mean(error > 0) * 100,
        "porcentaje_sobreestimacion": np.mean(error < 0) * 100,
        "porcentaje_error_cero": np.mean(error == 0) * 100,

        "correlacion": correlacion,

        "calibracion_intercepto": intercepto,
        "calibracion_pendiente": pendiente
    }


# ------------------------------------------------------------
# 3. CARGA Y VALIDACIÓN DE LOS DATOS
# ------------------------------------------------------------

print("\nDIAGNÓSTICO DE CALIBRACIÓN Y ERRORES")
print("=" * 105)

if not ARCHIVO_ENTRADA.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo: {ARCHIVO_ENTRADA}"
    )

df = pd.read_csv(ARCHIVO_ENTRADA)

columnas_requeridas = [
    "fecha_objetivo",
    "horizonte_horas",
    "particion",
    "modelo",
    VARIABLE_REAL,
    VARIABLE_PREDICHA
]

faltantes = [
    columna
    for columna in columnas_requeridas
    if columna not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas requeridas: {faltantes}"
    )

df["fecha_objetivo"] = pd.to_datetime(
    df["fecha_objetivo"],
    errors="coerce",
    utc=True
)

df[VARIABLE_REAL] = pd.to_numeric(
    df[VARIABLE_REAL],
    errors="coerce"
)

df[VARIABLE_PREDICHA] = pd.to_numeric(
    df[VARIABLE_PREDICHA],
    errors="coerce"
)

df = df[
    (df["particion"] == PARTICION)
    & (df["modelo"].isin(MODELOS))
    & (df["horizonte_horas"].isin(HORIZONTES))
].copy()

df = df.dropna(
    subset=[
        "fecha_objetivo",
        "horizonte_horas",
        "modelo",
        VARIABLE_REAL,
        VARIABLE_PREDICHA
    ]
)

print(f"Observaciones individuales: {len(df)}")


# ------------------------------------------------------------
# 4. EMPAREJAMIENTO DE MODELOS
# ------------------------------------------------------------

resultados = []

for horizonte in HORIZONTES:

    print("\n" + "=" * 105)
    print(f"HORIZONTE: {horizonte} h")
    print("=" * 105)

    datos_h = df[
        df["horizonte_horas"] == horizonte
    ].copy()

    # Separar predicciones de cada modelo.
    mlp = datos_h[
        datos_h["modelo"] == MODELOS[0]
    ][
        ["fecha_objetivo", VARIABLE_REAL, VARIABLE_PREDICHA]
    ].rename(
        columns={
            VARIABLE_REAL: "o3_real_mlp",
            VARIABLE_PREDICHA: "o3_predicho_mlp"
        }
    )

    fisico = datos_h[
        datos_h["modelo"] == MODELOS[1]
    ][
        ["fecha_objetivo", VARIABLE_REAL, VARIABLE_PREDICHA]
    ].rename(
        columns={
            VARIABLE_REAL: "o3_real_fisico",
            VARIABLE_PREDICHA: "o3_predicho_fisico"
        }
    )

    # Comprobar que cada modelo tiene una sola predicción
    # por fecha y que las fechas pueden emparejarse.
    if mlp["fecha_objetivo"].duplicated().any():
        raise ValueError(
            f"Fechas duplicadas en MLP para horizonte {horizonte}."
        )

    if fisico["fecha_objetivo"].duplicated().any():
        raise ValueError(
            f"Fechas duplicadas en modelo físico para horizonte {horizonte}."
        )

    comunes = mlp.merge(
        fisico,
        on="fecha_objetivo",
        how="inner",
        validate="one_to_one"
    )

    # Verificar que ambos modelos utilizan exactamente
    # la misma concentración observada.
    diferencia_real = np.abs(
        comunes["o3_real_mlp"]
        - comunes["o3_real_fisico"]
    )

    if (diferencia_real > 1e-8).any():
        raise ValueError(
            f"Los valores reales no coinciden en horizonte {horizonte}."
        )

    comunes["o3_real"] = comunes["o3_real_mlp"]

    print(f"Fechas comunes: {len(comunes)}")

    if len(comunes) == 0:
        print("Sin observaciones comunes.")
        continue

    # --------------------------------------------------------
    # 5. ESTRATIFICACIÓN EN CINCO CUANTILES
    # --------------------------------------------------------

    cortes = np.quantile(
        comunes["o3_real"].to_numpy(),
        [0.20, 0.40, 0.60, 0.80]
    )

    print("\nPuntos de corte:")
    for i, corte in enumerate(cortes, start=20):
        if i in [20, 40, 60, 80]:
            print(
                f"P{i}: {corte:.4f} µg/m³"
            )

    # La asignación mantiene juntos los valores idénticos.
    # Los valores exactamente iguales al punto de corte
    # permanecen en el estrato inferior.
    comunes["estrato"] = (
        np.searchsorted(
            cortes,
            comunes["o3_real"].to_numpy(),
            side="left"
        ) + 1
    )

    # --------------------------------------------------------
    # 6. MÉTRICAS POR MODELO Y ESTRATO
    # --------------------------------------------------------

    for estrato in range(1, 6):

        grupo = comunes[
            comunes["estrato"] == estrato
        ].copy()

        if grupo.empty:
            print(f"\nQ{estrato}: sin observaciones.")
            continue

        print("\n" + "-" * 85)
        print(
            f"Q{estrato} | "
            f"n={len(grupo)} | "
            f"Rango={grupo['o3_real'].min():.2f}–"
            f"{grupo['o3_real'].max():.2f} µg/m³"
        )
        print("-" * 85)

        # Preparar los datos de cada modelo.
        grupo_mlp = pd.DataFrame({
            VARIABLE_REAL: grupo["o3_real"],
            VARIABLE_PREDICHA: grupo["o3_predicho_mlp"]
        })

        grupo_fisico = pd.DataFrame({
            VARIABLE_REAL: grupo["o3_real"],
            VARIABLE_PREDICHA: grupo["o3_predicho_fisico"]
        })

        metricas_mlp = calcular_metricas(grupo_mlp)
        metricas_fisico = calcular_metricas(grupo_fisico)

        for nombre, metricas in [
            (MODELOS[0], metricas_mlp),
            (MODELOS[1], metricas_fisico)
        ]:

            registro = {
                "horizonte_horas": horizonte,
                "estrato": f"Q{estrato}",
                "modelo": nombre,
                "concentracion_min": grupo["o3_real"].min(),
                "concentracion_max": grupo["o3_real"].max()
            }

            registro.update(metricas)
            resultados.append(registro)

            print(f"\n{nombre}")
            print(
                f"  Sesgo:                  {metricas['sesgo']:.4f}"
            )
            print(
                f"  MAE:                    {metricas['mae']:.4f}"
            )
            print(
                f"  RMSE:                   {metricas['rmse']:.4f}"
            )
            print(
                f"  Desviación del error:   {metricas['desviacion_error']:.4f}"
            )
            print(
                f"  P90 error absoluto:     {metricas['p90_error_absoluto']:.4f}"
            )
            print(
                f"  Sesgo² / MSE:           {metricas['proporcion_mse_sesgo_pct']:.2f}%"
            )
            print(
                f"  Subestimaciones:        {metricas['porcentaje_subestimacion']:.2f}%"
            )
            print(
                f"  Sobreestimaciones:      {metricas['porcentaje_sobreestimacion']:.2f}%"
            )
            print(
                f"  Calibración pendiente:  {metricas['calibracion_pendiente']:.4f}"
            )
            print(
                f"  Calibración intercepto: {metricas['calibracion_intercepto']:.4f}"
            )

        # ----------------------------------------------------
        # 7. COMPARACIÓN ENTRE MODELOS
        # ----------------------------------------------------

        delta_mae = (
            metricas_mlp["mae"] - metricas_fisico["mae"]
        )

        delta_rmse = (
            metricas_mlp["rmse"] - metricas_fisico["rmse"]
        )

        delta_sesgo = (
            metricas_mlp["sesgo"] - metricas_fisico["sesgo"]
        )

        delta_desviacion = (
            metricas_mlp["desviacion_error"]
            - metricas_fisico["desviacion_error"]
        )

        delta_varianza = (
            metricas_mlp["varianza_error"]
            - metricas_fisico["varianza_error"]
        )

        delta_p90 = (
            metricas_mlp["p90_error_absoluto"]
            - metricas_fisico["p90_error_absoluto"]
        )

        delta_proporcion_sesgo = (
            metricas_mlp["proporcion_mse_sesgo_pct"]
            - metricas_fisico["proporcion_mse_sesgo_pct"]
        )

        print("\nCOMPARACIÓN (MLP - FÍSICO)")
        print(
            f"  Delta MAE:                 {delta_mae:+.4f}"
        )
        print(
            f"  Delta RMSE:                {delta_rmse:+.4f}"
        )
        print(
            f"  Delta sesgo:               {delta_sesgo:+.4f}"
        )
        print(
            f"  Delta desviación error:    {delta_desviacion:+.4f}"
        )
        print(
            f"  Delta varianza error:      {delta_varianza:+.4f}"
        )
        print(
            f"  Delta P90 error absoluto:  {delta_p90:+.4f}"
        )
        print(
            f"  Delta proporción sesgo:    {delta_proporcion_sesgo:+.2f} puntos porcentuales"
        )


# ------------------------------------------------------------
# 8. GUARDAR RESULTADOS
# ------------------------------------------------------------

if not resultados:
    raise ValueError(
        "No se generaron resultados. Revisa los datos de entrada."
    )

tabla = pd.DataFrame(resultados)

DIRECTORIO_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

tabla.to_csv(
    ARCHIVO_SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 105)
print("ANÁLISIS FINALIZADO")
print("=" * 105)

print(f"Archivo generado: {ARCHIVO_SALIDA}")

print("\nRESUMEN DE DESVIACIÓN DEL ERROR")

resumen = tabla.pivot(
    index="estrato",
    columns=["horizonte_horas", "modelo"],
    values="desviacion_error"
)

print(resumen.round(4).to_string())

print("\nRESUMEN DE CONTRIBUCIÓN DEL SESGO AL MSE (%)")

resumen_sesgo = tabla.pivot(
    index="estrato",
    columns=["horizonte_horas", "modelo"],
    values="proporcion_mse_sesgo_pct"
)

print(resumen_sesgo.round(2).to_string())

print("\n" + "=" * 105)
