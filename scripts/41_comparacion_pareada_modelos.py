
# ============================================================
# SCRIPT 41
# COMPARACION APAREADA ENTRE MLP Y MODELO NEURONAL FISICO
#
# Objetivos:
# 1. Comparar ambos modelos en las mismas fechas.
# 2. Calcular diferencias de error absoluto y cuadratico.
# 3. Identificar la frecuencia con la que cada modelo gana.
# 4. Examinar las diferencias por intervalos de O3 observado.
#
# Se utiliza exclusivamente la particion validation.
# No se modifican ni reentrenan los modelos.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. CONFIGURACION
# ------------------------------------------------------------

ARCHIVO_ENTRADA = Path(
    "outputs/tables/o3_pinn_predicciones_detalladas.csv"
)

DIRECTORIO_SALIDA = Path("outputs/tables")

ARCHIVO_SALIDA = (
    DIRECTORIO_SALIDA /
    "41_comparacion_pareada_modelos.csv"
)

MODELO_MLP = "MLP_con_meteorologia"
MODELO_FISICO = "Modelo_neuronal_fisico"

HORIZONTES = [1, 6, 24]
PARTICION = "validation"

ANCHO_INTERVALO = 20.0

# Tamanio minimo para considerar un intervalo
# con una cantidad razonable de observaciones.
N_MINIMO = 30


# ------------------------------------------------------------
# 2. FUNCIONES AUXILIARES
# ------------------------------------------------------------

def crear_intervalos(valores):
    """
    Crea intervalos absolutos de concentracion
    con la misma anchura para ambos modelos.
    """

    minimo = (
        np.floor(
            np.min(valores) / ANCHO_INTERVALO
        ) * ANCHO_INTERVALO
    )

    maximo = (
        np.ceil(
            np.max(valores) / ANCHO_INTERVALO
        ) * ANCHO_INTERVALO
    )

    limites = np.arange(
        minimo,
        maximo + ANCHO_INTERVALO,
        ANCHO_INTERVALO
    )

    if len(limites) < 2:
        limites = np.array([
            minimo,
            minimo + ANCHO_INTERVALO
        ])

    return limites


def calcular_comparacion(grupo):
    """
    Calcula las diferencias apareadas de errores.

    Delta positivo favorece al modelo fisico.
    Delta negativo favorece al MLP.
    """

    delta_absoluto = grupo[
        "delta_error_absoluto"
    ].to_numpy()

    delta_cuadratico = grupo[
        "delta_error_cuadratico"
    ].to_numpy()

    n = len(grupo)

    # Tolerancia numerica para clasificar empates.
    tolerancia = 1e-10

    victorias_fisico = np.sum(
        delta_absoluto > tolerancia
    )

    victorias_mlp = np.sum(
        delta_absoluto < -tolerancia
    )

    empates = n - victorias_fisico - victorias_mlp

    return {
        "observaciones": n,

        "delta_mae_apareado_medio": np.mean(
            delta_absoluto
        ),

        "delta_mae_apareado_mediana": np.median(
            delta_absoluto
        ),

        "delta_mae_p10": np.percentile(
            delta_absoluto,
            10
        ),

        "delta_mae_p90": np.percentile(
            delta_absoluto,
            90
        ),

        "proporcion_fisico_mejor_pct": (
            victorias_fisico / n * 100
        ),

        "proporcion_mlp_mejor_pct": (
            victorias_mlp / n * 100
        ),

        "proporcion_empate_pct": (
            empates / n * 100
        ),

        "delta_error_cuadratico_medio": np.mean(
            delta_cuadratico
        ),

        "delta_error_cuadratico_mediana": np.median(
            delta_cuadratico
        ),

        "proporcion_fisico_menor_error_cuadratico_pct": (
            np.mean(delta_cuadratico > tolerancia) * 100
        )
    }


# ------------------------------------------------------------
# 3. CARGA Y VALIDACION DE DATOS
# ------------------------------------------------------------

print("\nCOMPARACION APAREADA ENTRE MODELOS")
print("=" * 105)

if not ARCHIVO_ENTRADA.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo: {ARCHIVO_ENTRADA}"
    )

DIRECTORIO_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(
    ARCHIVO_ENTRADA
)

columnas_requeridas = [
    "fecha_objetivo",
    "horizonte_horas",
    "particion",
    "modelo",
    "o3_real",
    "o3_predicho"
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

df["o3_real"] = pd.to_numeric(
    df["o3_real"],
    errors="coerce"
)

df["o3_predicho"] = pd.to_numeric(
    df["o3_predicho"],
    errors="coerce"
)

df = df[
    (df["particion"] == PARTICION)
    & (
        df["modelo"].isin([
            MODELO_MLP,
            MODELO_FISICO
        ])
    )
    & (
        df["horizonte_horas"].isin(HORIZONTES)
    )
].copy()

df = df.dropna(
    subset=[
        "fecha_objetivo",
        "horizonte_horas",
        "modelo",
        "o3_real",
        "o3_predicho"
    ]
)

print(
    f"Observaciones individuales: {len(df)}"
)


# ------------------------------------------------------------
# 4. COMPARACION POR HORIZONTE
# ------------------------------------------------------------

resultados = []

for horizonte in HORIZONTES:

    print("\n" + "=" * 105)
    print(f"HORIZONTE: {horizonte} h")
    print("=" * 105)

    datos_h = df[
        df["horizonte_horas"] == horizonte
    ].copy()

    # Separar las predicciones del MLP.
    mlp = datos_h[
        datos_h["modelo"] == MODELO_MLP
    ][
        ["fecha_objetivo", "o3_real", "o3_predicho"]
    ].rename(
        columns={
            "o3_real": "o3_real_mlp",
            "o3_predicho": "o3_predicho_mlp"
        }
    )

    # Separar las predicciones del modelo fisico.
    fisico = datos_h[
        datos_h["modelo"] == MODELO_FISICO
    ][
        ["fecha_objetivo", "o3_real", "o3_predicho"]
    ].rename(
        columns={
            "o3_real": "o3_real_fisico",
            "o3_predicho": "o3_predicho_fisico"
        }
    )

    # Verificar fechas duplicadas.
    if mlp["fecha_objetivo"].duplicated().any():
        raise ValueError(
            f"Fechas duplicadas en MLP: horizonte {horizonte}"
        )

    if fisico["fecha_objetivo"].duplicated().any():
        raise ValueError(
            f"Fechas duplicadas en fisico: horizonte {horizonte}"
        )

    # Emparejar ambas predicciones por fecha.
    comunes = mlp.merge(
        fisico,
        on="fecha_objetivo",
        how="inner",
        validate="one_to_one"
    )

    if comunes.empty:
        print("No existen fechas comunes.")
        continue

    # Verificar que ambos modelos tengan
    # exactamente la misma observacion real.
    diferencia_real = np.abs(
        comunes["o3_real_mlp"]
        - comunes["o3_real_fisico"]
    )

    if (diferencia_real > 1e-8).any():
        raise ValueError(
            f"Los valores reales no coinciden: horizonte {horizonte}"
        )

    comunes["o3_real"] = comunes["o3_real_mlp"]

    print(
        f"Fechas emparejadas: {len(comunes)}"
    )

    # --------------------------------------------------------
    # 5. CALCULAR ERRORES APAREADOS
    # --------------------------------------------------------

    # Error = observacion - prediccion.
    # Positivo: subestimacion.
    # Negativo: sobreestimacion.

    comunes["error_mlp"] = (
        comunes["o3_real"]
        - comunes["o3_predicho_mlp"]
    )

    comunes["error_fisico"] = (
        comunes["o3_real"]
        - comunes["o3_predicho_fisico"]
    )

    comunes["error_absoluto_mlp"] = np.abs(
        comunes["error_mlp"]
    )

    comunes["error_absoluto_fisico"] = np.abs(
        comunes["error_fisico"]
    )

    comunes["error_cuadratico_mlp"] = (
        comunes["error_mlp"] ** 2
    )

    comunes["error_cuadratico_fisico"] = (
        comunes["error_fisico"] ** 2
    )

    # Delta positivo favorece al modelo fisico.
    comunes["delta_error_absoluto"] = (
        comunes["error_absoluto_mlp"]
        - comunes["error_absoluto_fisico"]
    )

    comunes["delta_error_cuadratico"] = (
        comunes["error_cuadratico_mlp"]
        - comunes["error_cuadratico_fisico"]
    )

    # --------------------------------------------------------
    # 6. COMPARACION GLOBAL APAREADA
    # --------------------------------------------------------

    resumen_global = calcular_comparacion(
        comunes
    )

    registro_global = {
        "horizonte_horas": horizonte,
        "tipo_analisis": "GLOBAL",
        "intervalo_inferior": np.nan,
        "intervalo_superior": np.nan,
        "intervalo": "GLOBAL"
    }

    registro_global.update(
        resumen_global
    )

    resultados.append(
        registro_global
    )

    print("\nCOMPARACION GLOBAL APAREADA")
    print("-" * 75)

    print(
        f"Delta MAE medio: "
        f"{resumen_global['delta_mae_apareado_medio']:+.4f}"
    )

    print(
        f"Delta MAE mediano: "
        f"{resumen_global['delta_mae_apareado_mediana']:+.4f}"
    )

    print(
        f"Percentil 10: "
        f"{resumen_global['delta_mae_p10']:+.4f}"
    )

    print(
        f"Percentil 90: "
        f"{resumen_global['delta_mae_p90']:+.4f}"
    )

    print(
        f"Fisico con menor error absoluto: "
        f"{resumen_global['proporcion_fisico_mejor_pct']:.2f}%"
    )

    print(
        f"MLP con menor error absoluto: "
        f"{resumen_global['proporcion_mlp_mejor_pct']:.2f}%"
    )

    print(
        f"Empates: "
        f"{resumen_global['proporcion_empate_pct']:.2f}%"
    )

    print(
        f"Delta error cuadratico medio: "
        f"{resumen_global['delta_error_cuadratico_medio']:+.4f}"
    )

    # --------------------------------------------------------
    # 7. INTERVALOS ABSOLUTOS DE CONCENTRACION
    # --------------------------------------------------------

    limites = crear_intervalos(
        comunes["o3_real"].to_numpy()
    )

    comunes["intervalo"] = pd.cut(
        comunes["o3_real"],
        bins=limites,
        right=False,
        include_lowest=True
    )

    print("\nCOMPARACION POR INTERVALOS")
    print("-" * 105)

    print(
        "Intervalo       n       Delta MAE       "
        "Fisico mejor (%)       Delta error cuadratico"
    )

    for intervalo in comunes["intervalo"].cat.categories:

        grupo = comunes[
            comunes["intervalo"] == intervalo
        ].copy()

        if grupo.empty:
            continue

        inferior = intervalo.left
        superior = intervalo.right

        resumen = calcular_comparacion(
            grupo
        )

        registro = {
            "horizonte_horas": horizonte,
            "tipo_analisis": "CONDICIONAL",
            "intervalo_inferior": inferior,
            "intervalo_superior": superior,
            "intervalo": (
                f"[{inferior:.0f}, {superior:.0f})"
            )
        }

        registro.update(
            resumen
        )

        resultados.append(
            registro
        )

        # Identificar intervalos con pocas observaciones.
        marca = (
            ""
            if len(grupo) >= N_MINIMO
            else " *"
        )

        print(
            f"[{inferior:3.0f}, {superior:3.0f})"
            f"   {len(grupo):4d}"
            f"      {resumen['delta_mae_apareado_medio']:+8.4f}"
            f"             {resumen['proporcion_fisico_mejor_pct']:7.2f}%"
            f"              {resumen['delta_error_cuadratico_medio']:+10.4f}"
            f"{marca}"
        )

    print(
        f"\n* Intervalos con menos de {N_MINIMO} observaciones."
    )


# ------------------------------------------------------------
# 8. GUARDAR RESULTADOS
# ------------------------------------------------------------

if not resultados:
    raise ValueError(
        "No se generaron resultados de comparacion."
    )

tabla = pd.DataFrame(
    resultados
)

tabla.to_csv(
    ARCHIVO_SALIDA,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 9. RESUMEN FINAL
# ------------------------------------------------------------

print("\n" + "=" * 105)
print("RESUMEN FINAL DE COMPARACION APAREADA")
print("=" * 105)

for horizonte in HORIZONTES:

    grupo = tabla[
        (tabla["horizonte_horas"] == horizonte)
        & (tabla["tipo_analisis"] == "GLOBAL")
    ]

    if grupo.empty:
        continue

    fila = grupo.iloc[0]

    print(f"\nHorizonte: {horizonte} h")

    print(
        f"Observaciones: {int(fila['observaciones'])}"
    )

    print(
        f"Delta MAE: "
        f"{fila['delta_mae_apareado_medio']:+.4f}"
    )

    print(
        f"Proporcion de victorias del fisico: "
        f"{fila['proporcion_fisico_mejor_pct']:.2f}%"
    )

    print(
        f"Proporcion de victorias del MLP: "
        f"{fila['proporcion_mlp_mejor_pct']:.2f}%"
    )

print("\n" + "=" * 105)
print("ANALISIS FINALIZADO")
print("=" * 105)

print(
    f"\nArchivo generado: {ARCHIVO_SALIDA}"
)
