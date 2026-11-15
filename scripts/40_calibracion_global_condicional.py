
# ============================================================
# SCRIPT 40
# CALIBRACION GLOBAL Y ANALISIS CONDICIONAL DE ERRORES
#
# Objetivos:
# 1. Evaluar la calibracion global de ambos modelos.
# 2. Examinar el error segun el nivel observado de O3.
# 3. Comparar los modelos en intervalos absolutos comunes.
# 4. Generar graficos de calibracion y residuos.
#
# Se utiliza exclusivamente la particion validation.
# No se modifican ni reentrenan los modelos.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. CONFIGURACION
# ------------------------------------------------------------

ARCHIVO_ENTRADA = Path(
    "outputs/tables/o3_pinn_predicciones_detalladas.csv"
)

DIRECTORIO_SALIDA = Path("outputs/tables")
DIRECTORIO_GRAFICOS = Path("outputs/figures/40_calibracion")

ARCHIVO_GLOBAL = (
    DIRECTORIO_SALIDA /
    "40_calibracion_global.csv"
)

ARCHIVO_CONDICIONAL = (
    DIRECTORIO_SALIDA /
    "40_errores_condicionales.csv"
)

MODELOS = [
    "MLP_con_meteorologia",
    "Modelo_neuronal_fisico"
]

HORIZONTES = [1, 6, 24]

PARTICION = "validation"

VARIABLE_REAL = "o3_real"
VARIABLE_PREDICHA = "o3_predicho"

# Anchura de los intervalos absolutos de concentracion.
ANCHO_INTERVALO = 20.0

# Unidades de concentracion.
UNIDAD = "µg/m³"


# ------------------------------------------------------------
# 2. FUNCIONES AUXILIARES
# ------------------------------------------------------------

def calcular_metricas(real, predicho):
    """
    Calcula metricas globales de prediccion y calibracion.

    El residuo se define como:
        error = real - predicho

    Por tanto:
        error positivo = subestimacion
        error negativo = sobreestimacion
    """

    real = np.asarray(real, dtype=float)
    predicho = np.asarray(predicho, dtype=float)

    error = real - predicho
    error_absoluto = np.abs(error)

    sesgo = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))

    # Regresion de calibracion:
    # real = intercepto + pendiente * prediccion.
    #
    # Calibracion ideal:
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

        correlacion = np.corrcoef(
            real,
            predicho
        )[0, 1]

    else:
        pendiente = np.nan
        intercepto = np.nan
        correlacion = np.nan

    return {
        "observaciones": len(real),

        "o3_real_medio": np.mean(real),
        "o3_predicho_medio": np.mean(predicho),

        "sesgo": sesgo,
        "sesgo_absoluto": abs(sesgo),

        "mae": np.mean(error_absoluto),
        "rmse": rmse,

        "desviacion_error": np.std(
            error,
            ddof=0
        ),

        "mediana_error_absoluto": np.median(
            error_absoluto
        ),

        "p90_error_absoluto": np.percentile(
            error_absoluto,
            90
        ),

        "porcentaje_subestimacion": np.mean(
            error > 0
        ) * 100,

        "porcentaje_sobreestimacion": np.mean(
            error < 0
        ) * 100,

        "correlacion": correlacion,

        "calibracion_pendiente": pendiente,
        "calibracion_intercepto": intercepto
    }


def calcular_regresion(real, predicho):
    """
    Obtiene la recta de regresion:
        real = intercepto + pendiente * predicho.
    """

    if (
        len(real) < 3
        or np.std(predicho) == 0
        or np.std(real) == 0
    ):
        return np.nan, np.nan

    pendiente, intercepto = np.polyfit(
        predicho,
        real,
        1
    )

    return pendiente, intercepto


def crear_intervalos(valores):
    """
    Genera intervalos absolutos de anchura fija.

    Los mismos limites se aplican a ambos modelos
    dentro de cada horizonte.
    """

    minimo = np.floor(
        np.min(valores) / ANCHO_INTERVALO
    ) * ANCHO_INTERVALO

    maximo = np.ceil(
        np.max(valores) / ANCHO_INTERVALO
    ) * ANCHO_INTERVALO

    # Asegurar que el ultimo valor quede incluido.
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


# ------------------------------------------------------------
# 3. PREPARACION DE DIRECTORIOS Y DATOS
# ------------------------------------------------------------

print("\nCALIBRACION GLOBAL Y ANALISIS CONDICIONAL")
print("=" * 105)

if not ARCHIVO_ENTRADA.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo: {ARCHIVO_ENTRADA}"
    )

DIRECTORIO_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

DIRECTORIO_GRAFICOS.mkdir(
    parents=True,
    exist_ok=True
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

print(
    f"Observaciones individuales: {len(df)}"
)


# ------------------------------------------------------------
# 4. ANALISIS POR HORIZONTE
# ------------------------------------------------------------

resultados_globales = []
resultados_condicionales = []

for horizonte in HORIZONTES:

    print("\n" + "=" * 105)
    print(f"HORIZONTE: {horizonte} h")
    print("=" * 105)

    datos_h = df[
        df["horizonte_horas"] == horizonte
    ].copy()

    # Separar las predicciones de cada modelo.
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

    # Evitar emparejamientos ambiguos.
    if mlp["fecha_objetivo"].duplicated().any():
        raise ValueError(
            f"Fechas duplicadas en MLP: horizonte {horizonte}"
        )

    if fisico["fecha_objetivo"].duplicated().any():
        raise ValueError(
            f"Fechas duplicadas en modelo fisico: horizonte {horizonte}"
        )

    # Emparejar por fecha.
    comunes = mlp.merge(
        fisico,
        on="fecha_objetivo",
        how="inner",
        validate="one_to_one"
    )

    # Ambos modelos deben utilizar el mismo valor real.
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
        f"Fechas comunes: {len(comunes)}"
    )

    if comunes.empty:
        print("No hay fechas comunes.")
        continue

    # --------------------------------------------------------
    # 5. CALIBRACION GLOBAL
    # --------------------------------------------------------

    metricas_modelos = {}

    for nombre, columna_prediccion in [
        (MODELOS[0], "o3_predicho_mlp"),
        (MODELOS[1], "o3_predicho_fisico")
    ]:

        real = comunes["o3_real"].to_numpy()

        predicho = comunes[
            columna_prediccion
        ].to_numpy()

        metricas = calcular_metricas(
            real,
            predicho
        )

        metricas_modelos[nombre] = metricas

        registro = {
            "horizonte_horas": horizonte,
            "modelo": nombre
        }

        registro.update(metricas)
        resultados_globales.append(registro)

        print("\n" + "-" * 75)
        print(nombre)
        print("-" * 75)

        print(
            f"Sesgo:                  {metricas['sesgo']:.4f}"
        )
        print(
            f"MAE:                    {metricas['mae']:.4f}"
        )
        print(
            f"RMSE:                   {metricas['rmse']:.4f}"
        )
        print(
            f"Desviacion del error:   {metricas['desviacion_error']:.4f}"
        )
        print(
            f"Correlacion:            {metricas['correlacion']:.4f}"
        )
        print(
            f"Pendiente calibracion:  {metricas['calibracion_pendiente']:.4f}"
        )
        print(
            f"Intercepto calibracion: {metricas['calibracion_intercepto']:.4f}"
        )

    # Comparacion global.
    delta_mae = (
        metricas_modelos[MODELOS[0]]["mae"]
        - metricas_modelos[MODELOS[1]]["mae"]
    )

    delta_rmse = (
        metricas_modelos[MODELOS[0]]["rmse"]
        - metricas_modelos[MODELOS[1]]["rmse"]
    )

    print("\nCOMPARACION GLOBAL (MLP - FISICO)")
    print(
        f"Delta MAE:  {delta_mae:+.4f}"
    )
    print(
        f"Delta RMSE: {delta_rmse:+.4f}"
    )

    # --------------------------------------------------------
    # 6. INTERVALOS ABSOLUTOS DE CONCENTRACION
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

    print("\nINTERVALOS DE CONCENTRACION")
    print(
        f"Anchura de intervalo: {ANCHO_INTERVALO:.1f} {UNIDAD}"
    )

    # --------------------------------------------------------
    # 7. ERRORES CONDICIONALES
    # --------------------------------------------------------

    for intervalo in comunes["intervalo"].cat.categories:

        grupo = comunes[
            comunes["intervalo"] == intervalo
        ].copy()

        if grupo.empty:
            continue

        inferior = intervalo.left
        superior = intervalo.right

        print("\n" + "-" * 85)
        print(
            f"INTERVALO: [{inferior:.0f}, {superior:.0f}) {UNIDAD}"
        )
        print(
            f"Observaciones: {len(grupo)}"
        )
        print("-" * 85)

        for nombre, columna_prediccion in [
            (MODELOS[0], "o3_predicho_mlp"),
            (MODELOS[1], "o3_predicho_fisico")
        ]:

            metricas = calcular_metricas(
                grupo["o3_real"].to_numpy(),
                grupo[columna_prediccion].to_numpy()
            )

            registro = {
                "horizonte_horas": horizonte,
                "intervalo_inferior": inferior,
                "intervalo_superior": superior,
                "intervalo": (
                    f"[{inferior:.0f}, {superior:.0f})"
                ),
                "modelo": nombre
            }

            registro.update(metricas)
            resultados_condicionales.append(registro)

            print(f"\n{nombre}")
            print(
                f"  O3 real medio:       {metricas['o3_real_medio']:.4f}"
            )
            print(
                f"  O3 predicho medio:   {metricas['o3_predicho_medio']:.4f}"
            )
            print(
                f"  Sesgo:               {metricas['sesgo']:.4f}"
            )
            print(
                f"  MAE:                 {metricas['mae']:.4f}"
            )
            print(
                f"  RMSE:                {metricas['rmse']:.4f}"
            )
            print(
                f"  Desviacion error:    {metricas['desviacion_error']:.4f}"
            )
            print(
                f"  P90 error absoluto:  {metricas['p90_error_absoluto']:.4f}"
            )

    # --------------------------------------------------------
    # 8. GRAFICO DE CALIBRACION
    # --------------------------------------------------------

    real = comunes["o3_real"].to_numpy()

    pred_mlp = comunes[
        "o3_predicho_mlp"
    ].to_numpy()

    pred_fisico = comunes[
        "o3_predicho_fisico"
    ].to_numpy()

    minimo = min(
        np.min(real),
        np.min(pred_mlp),
        np.min(pred_fisico)
    )

    maximo = max(
        np.max(real),
        np.max(pred_mlp),
        np.max(pred_fisico)
    )

    margen = (maximo - minimo) * 0.04

    limite_inferior = minimo - margen
    limite_superior = maximo + margen

    figura, eje = plt.subplots(
        figsize=(9, 8)
    )

    # Predicciones del MLP.
    eje.scatter(
        pred_mlp,
        real,
        alpha=0.25,
        s=16,
        label="MLP con meteorologia"
    )

    # Predicciones del modelo fisico.
    eje.scatter(
        pred_fisico,
        real,
        alpha=0.25,
        s=16,
        label="Modelo neuronal fisico"
    )

    # Linea ideal de calibracion.
    eje.plot(
        [limite_inferior, limite_superior],
        [limite_inferior, limite_superior],
        linestyle="--",
        linewidth=2,
        label="Calibracion ideal (y = x)"
    )

    # Rectas de calibracion global.
    for nombre, predicho in [
        ("MLP", pred_mlp),
        ("Fisico", pred_fisico)
    ]:

        pendiente, intercepto = calcular_regresion(
            real,
            predicho
        )

        if np.isfinite(pendiente):
            x_linea = np.linspace(
                limite_inferior,
                limite_superior,
                100
            )

            y_linea = (
                intercepto
                + pendiente * x_linea
            )

            eje.plot(
                x_linea,
                y_linea,
                linewidth=2,
                label=(
                    f"Ajuste {nombre}: "
                    f"pendiente={pendiente:.3f}"
                )
            )

    eje.set_title(
        f"Calibracion global - Horizonte {horizonte} h"
    )

    eje.set_xlabel(
        f"O3 predicho ({UNIDAD})"
    )

    eje.set_ylabel(
        f"O3 observado ({UNIDAD})"
    )

    eje.set_xlim(
        limite_inferior,
        limite_superior
    )

    eje.set_ylim(
        limite_inferior,
        limite_superior
    )

    eje.grid(
        alpha=0.25
    )

    eje.legend(
        fontsize=8
    )

    figura.tight_layout()

    archivo_calibracion = (
        DIRECTORIO_GRAFICOS /
        f"40_calibracion_{horizonte}h.png"
    )

    figura.savefig(
        archivo_calibracion,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figura)

    # --------------------------------------------------------
    # 9. GRAFICO DE RESIDUOS
    # --------------------------------------------------------

    error_mlp = real - pred_mlp
    error_fisico = real - pred_fisico

    figura, eje = plt.subplots(
        figsize=(10, 7)
    )

    eje.scatter(
        real,
        error_mlp,
        alpha=0.22,
        s=15,
        label="MLP con meteorologia"
    )

    eje.scatter(
        real,
        error_fisico,
        alpha=0.22,
        s=15,
        label="Modelo neuronal fisico"
    )

    # Referencia de error cero.
    eje.axhline(
        0,
        linestyle="--",
        linewidth=1.5,
        label="Error cero"
    )

    # Promedio del residuo dentro de cada intervalo.
    for nombre, columna_prediccion in [
        ("MLP", "o3_predicho_mlp"),
        ("Fisico", "o3_predicho_fisico")
    ]:

        columna_error = (
            "error_mlp"
            if nombre == "MLP"
            else "error_fisico"
        )

        comunes[columna_error] = (
            comunes["o3_real"]
            - comunes[columna_prediccion]
        )

        resumen_intervalos = (
            comunes.groupby(
                "intervalo",
                observed=True
            )
            .agg(
                o3_medio=("o3_real", "mean"),
                error_medio=(columna_error, "mean")
            )
            .reset_index()
        )

        eje.plot(
            resumen_intervalos["o3_medio"],
            resumen_intervalos["error_medio"],
            marker="o",
            linewidth=2.5,
            markersize=6,
            label=f"Error medio por intervalo - {nombre}"
        )

    eje.set_title(
        f"Residuos frente a concentracion observada - {horizonte} h"
    )

    eje.set_xlabel(
        f"O3 observado ({UNIDAD})"
    )

    eje.set_ylabel(
        f"Error (observado - predicho) ({UNIDAD})"
    )

    eje.grid(
        alpha=0.25
    )

    eje.legend(
        fontsize=8
    )

    figura.tight_layout()

    archivo_residuos = (
        DIRECTORIO_GRAFICOS /
        f"40_residuos_{horizonte}h.png"
    )

    figura.savefig(
        archivo_residuos,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figura)

    print("\nGraficos guardados:")
    print(f"  {archivo_calibracion}")
    print(f"  {archivo_residuos}")


# ------------------------------------------------------------
# 10. GUARDAR TABLAS
# ------------------------------------------------------------

if not resultados_globales:
    raise ValueError(
        "No se generaron metricas globales."
    )

if not resultados_condicionales:
    raise ValueError(
        "No se generaron metricas condicionales."
    )

tabla_global = pd.DataFrame(
    resultados_globales
)

tabla_condicional = pd.DataFrame(
    resultados_condicionales
)

tabla_global.to_csv(
    ARCHIVO_GLOBAL,
    index=False,
    encoding="utf-8-sig"
)

tabla_condicional.to_csv(
    ARCHIVO_CONDICIONAL,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 11. RESUMEN COMPARATIVO GLOBAL
# ------------------------------------------------------------

print("\n" + "=" * 105)
print("RESUMEN GLOBAL")
print("=" * 105)

for horizonte in HORIZONTES:

    grupo = tabla_global[
        tabla_global["horizonte_horas"] == horizonte
    ]

    if len(grupo) != 2:
        continue

    mlp = grupo[
        grupo["modelo"] == MODELOS[0]
    ].iloc[0]

    fisico = grupo[
        grupo["modelo"] == MODELOS[1]
    ].iloc[0]

    print(f"\nHORIZONTE: {horizonte} h")

    print(
        f"Delta MAE (MLP - fisico): "
        f"{mlp['mae'] - fisico['mae']:+.4f}"
    )

    print(
        f"Delta RMSE (MLP - fisico): "
        f"{mlp['rmse'] - fisico['rmse']:+.4f}"
    )

    print(
        f"Delta sesgo absoluto: "
        f"{mlp['sesgo_absoluto'] - fisico['sesgo_absoluto']:+.4f}"
    )

    print(
        f"Delta desviacion error: "
        f"{mlp['desviacion_error'] - fisico['desviacion_error']:+.4f}"
    )


# ------------------------------------------------------------
# 12. RESUMEN DE ARCHIVOS
# ------------------------------------------------------------

print("\n" + "=" * 105)
print("ANALISIS FINALIZADO")
print("=" * 105)

print("\nArchivos generados:")

print(f"Calibracion global: {ARCHIVO_GLOBAL}")
print(f"Errores condicionales: {ARCHIVO_CONDICIONAL}")

print("\nGraficos:")
for horizonte in HORIZONTES:
    print(
        DIRECTORIO_GRAFICOS /
        f"40_calibracion_{horizonte}h.png"
    )

    print(
        DIRECTORIO_GRAFICOS /
        f"40_residuos_{horizonte}h.png"
    )

print("\n" + "=" * 105)
