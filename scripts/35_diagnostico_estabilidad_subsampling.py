
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = "outputs/tables/o3_pinn_predicciones_detalladas.csv"

SALIDA_RESUMEN = (
    "outputs/tables/35_resumen_estabilidad_subsampling.csv"
)

SALIDA_SEGMENTOS = (
    "outputs/tables/35_diagnostico_segmentos_subsampling.csv"
)

MLP = "MLP_con_meteorologia"
FISICO = "Modelo_neuronal_fisico"

BLOQUES_POR_HORIZONTE = {
    1: [2, 4, 6],
    6: [2, 4, 6],
    24: [6, 8, 12]
}

# ============================================================
# CONSTRUCCIÓN DE SEGMENTOS TEMPORALES
# ============================================================

def construir_segmentos(grupo):
    """
    Construye segmentos de observaciones horarias consecutivas.

    Solo conserva fechas con datos disponibles para ambos modelos.
    Un hueco temporal inicia un segmento nuevo.
    """

    matriz = grupo.pivot(
        index="fecha_objetivo",
        columns="modelo",
        values=[
            "error_absoluto",
            "error_cuadratico"
        ]
    )

    columnas = [
        (metrica, modelo)
        for metrica in [
            "error_absoluto",
            "error_cuadratico"
        ]
        for modelo in [MLP, FISICO]
    ]

    matriz = (
        matriz[columnas]
        .dropna()
        .sort_index()
    )

    diferencias = (
        matriz.index.to_series()
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    # Cada cambio distinto de una hora inicia otro segmento.
    identificadores = diferencias.ne(1).cumsum()

    segmentos = []

    for identificador, segmento in matriz.groupby(
        identificadores.to_numpy(),
        sort=False
    ):

        segmentos.append({
            "id_segmento": len(segmentos) + 1,
            "fecha_inicio": segmento.index.min(),
            "fecha_fin": segmento.index.max(),
            "mae_mlp": segmento[
                ("error_absoluto", MLP)
            ].to_numpy(dtype=float),
            "mae_fisico": segmento[
                ("error_absoluto", FISICO)
            ].to_numpy(dtype=float),
            "mse_mlp": segmento[
                ("error_cuadratico", MLP)
            ].to_numpy(dtype=float),
            "mse_fisico": segmento[
                ("error_cuadratico", FISICO)
            ].to_numpy(dtype=float)
        })

    return segmentos


# ============================================================
# MÉTRICAS SOBRE VARIOS SEGMENTOS
# ============================================================

def calcular_metricas(segmentos):
    """
    Calcula MAE y RMSE agrupando todas las observaciones.

    La diferencia se define como:

        MLP - modelo físico

    Una diferencia positiva favorece al modelo físico.
    """

    if not segmentos:
        return None

    n = sum(
        len(s["mae_mlp"])
        for s in segmentos
    )

    if n == 0:
        return None

    suma_mae_mlp = sum(
        np.sum(s["mae_mlp"])
        for s in segmentos
    )

    suma_mae_fisico = sum(
        np.sum(s["mae_fisico"])
        for s in segmentos
    )

    suma_mse_mlp = sum(
        np.sum(s["mse_mlp"])
        for s in segmentos
    )

    suma_mse_fisico = sum(
        np.sum(s["mse_fisico"])
        for s in segmentos
    )

    mae_mlp = suma_mae_mlp / n
    mae_fisico = suma_mae_fisico / n

    rmse_mlp = np.sqrt(suma_mse_mlp / n)
    rmse_fisico = np.sqrt(suma_mse_fisico / n)

    return {
        "n": n,
        "mae_mlp": mae_mlp,
        "mae_fisico": mae_fisico,
        "delta_mae": mae_mlp - mae_fisico,
        "rmse_mlp": rmse_mlp,
        "rmse_fisico": rmse_fisico,
        "delta_rmse": rmse_mlp - rmse_fisico
    }


# ============================================================
# MÉTRICAS DE UNA VENTANA
# ============================================================

def calcular_metricas_ventana(ventana):
    """
    Calcula la diferencia de MAE y RMSE en una ventana.

    RMSE se calcula como la raíz del promedio del error
    cuadrático, no como el promedio de errores absolutos.
    """

    mae_mlp = np.mean(ventana["mae_mlp"])
    mae_fisico = np.mean(ventana["mae_fisico"])

    rmse_mlp = np.sqrt(
        np.mean(ventana["mse_mlp"])
    )

    rmse_fisico = np.sqrt(
        np.mean(ventana["mse_fisico"])
    )

    return (
        mae_mlp - mae_fisico,
        rmse_mlp - rmse_fisico
    )


# ============================================================
# GENERACIÓN DE VENTANAS Y PESOS DE SUPERPOSICIÓN
# ============================================================

def analizar_ventanas(segmentos, longitud):
    """
    Genera todas las ventanas móviles completas.

    Además, cuenta cuántas ventanas incluyen cada observación.
    Esto permite medir el efecto de la superposición.
    """

    deltas_mae = []
    deltas_rmse = []

    # Acumuladores para calcular la ponderación inducida
    # por la cantidad de ventanas que contienen cada punto.
    suma_diferencias_mae = 0.0
    suma_pesos = 0

    ventanas_por_segmento = {}

    for segmento in segmentos:

        n = len(segmento["mae_mlp"])

        if n < longitud:
            continue

        # Diferencias individuales de error absoluto.
        diferencias_mae = (
            segmento["mae_mlp"]
            - segmento["mae_fisico"]
        )

        # Diferencias individuales de error cuadrático.
        diferencias_mse = (
            segmento["mse_mlp"]
            - segmento["mse_fisico"]
        )

        # Número de ventanas que contienen cada observación.
        pesos = np.zeros(n, dtype=int)

        ventanas_segmento = 0

        for inicio in range(n - longitud + 1):

            fin = inicio + longitud

            ventana = {
                "mae_mlp": segmento["mae_mlp"][inicio:fin],
                "mae_fisico": segmento["mae_fisico"][inicio:fin],
                "mse_mlp": segmento["mse_mlp"][inicio:fin],
                "mse_fisico": segmento["mse_fisico"][inicio:fin]
            }

            delta_mae, delta_rmse = (
                calcular_metricas_ventana(ventana)
            )

            deltas_mae.append(delta_mae)
            deltas_rmse.append(delta_rmse)

            # Registrar qué observaciones aparecen en esta ventana.
            pesos[inicio:fin] += 1

            ventanas_segmento += 1

        # La media de las ventanas equivale a una media
        # ponderada de las diferencias individuales.
        suma_diferencias_mae += np.sum(
            diferencias_mae * pesos
        )

        suma_pesos += np.sum(pesos)

        ventanas_por_segmento[
            segmento["id_segmento"]
        ] = ventanas_segmento

    if suma_pesos == 0:
        media_ponderada_mae = np.nan
    else:
        media_ponderada_mae = (
            suma_diferencias_mae / suma_pesos
        )

    return {
        "deltas_mae": np.asarray(deltas_mae),
        "deltas_rmse": np.asarray(deltas_rmse),
        "media_ponderada_mae": media_ponderada_mae,
        "ventanas_por_segmento": ventanas_por_segmento
    }


# ============================================================
# ESTADÍSTICAS DESCRIPTIVAS DE LAS VENTANAS
# ============================================================

def estadisticas_distribucion(valores, prefijo):
    """
    Resume la distribución de diferencias entre ventanas.
    No presupone que las ventanas sean independientes.
    """

    serie = pd.Series(valores, dtype=float)

    return {
        f"{prefijo}_media": serie.mean(),
        f"{prefijo}_mediana": serie.median(),
        f"{prefijo}_desviacion": serie.std(ddof=1),
        f"{prefijo}_p05": serie.quantile(0.05),
        f"{prefijo}_p25": serie.quantile(0.25),
        f"{prefijo}_p75": serie.quantile(0.75),
        f"{prefijo}_p95": serie.quantile(0.95),
        f"{prefijo}_asimetria": serie.skew()
    }


# ============================================================
# DIAGNÓSTICO DE INFLUENCIA DE SEGMENTOS
# ============================================================

def diagnosticar_influencia(
    segmentos,
    segmentos_elegibles,
    horizonte,
    bloque,
    observado
):
    """
    Evalúa cuánto cambia la diferencia global al retirar
    individualmente cada segmento.

    La influencia se mide como:

        diferencia sin segmento - diferencia completa

    No es una prueba de independencia ni de significación.
    """

    resultados = []

    # Suma absoluta de las contribuciones por segmento.
    contribuciones_absolutas = []

    for segmento in segmentos_elegibles:

        diferencia = (
            np.sum(segmento["mae_mlp"])
            - np.sum(segmento["mae_fisico"])
        )

        contribuciones_absolutas.append(
            abs(diferencia)
        )

    suma_contribuciones = sum(
        contribuciones_absolutas
    )

    for segmento in segmentos:

        n = len(segmento["mae_mlp"])

        elegible = n >= bloque

        registro = {
            "horizonte": horizonte,
            "bloque_horas": bloque,
            "id_segmento": segmento["id_segmento"],
            "fecha_inicio": segmento["fecha_inicio"],
            "fecha_fin": segmento["fecha_fin"],
            "longitud_segmento": n,
            "elegible": elegible
        }

        if not elegible:
            resultados.append(registro)
            continue

        metricas_segmento = calcular_metricas(
            [segmento]
        )

        restantes = [
            s for s in segmentos_elegibles
            if s["id_segmento"] != segmento["id_segmento"]
        ]

        metricas_sin_segmento = calcular_metricas(
            restantes
        )

        contribucion_mae = (
            np.sum(segmento["mae_mlp"])
            - np.sum(segmento["mae_fisico"])
        )

        proporcion_contribucion = (
            abs(contribucion_mae) / suma_contribuciones
            if suma_contribuciones > 0
            else np.nan
        )

        registro.update({
            "delta_mae_segmento":
                metricas_segmento["delta_mae"],

            "delta_rmse_segmento":
                metricas_segmento["delta_rmse"],

            "delta_mae_sin_segmento":
                metricas_sin_segmento["delta_mae"]
                if metricas_sin_segmento else np.nan,

            "delta_rmse_sin_segmento":
                metricas_sin_segmento["delta_rmse"]
                if metricas_sin_segmento else np.nan,

            "cambio_delta_mae":
                metricas_sin_segmento["delta_mae"]
                - observado["delta_mae"]
                if metricas_sin_segmento else np.nan,

            "cambio_delta_rmse":
                metricas_sin_segmento["delta_rmse"]
                - observado["delta_rmse"]
                if metricas_sin_segmento else np.nan,

            "proporcion_contribucion_absoluta_mae":
                proporcion_contribucion
        })

        resultados.append(registro)

    return resultados


# ============================================================
# CARGA Y VALIDACIÓN
# ============================================================

df = pd.read_csv(ARCHIVO)

df["fecha_objetivo"] = pd.to_datetime(
    df["fecha_objetivo"],
    utc=True
)

df = df[
    df["particion"] == "validation"
].copy()

# Comprobar que los modelos necesarios están presentes.
modelos_disponibles = set(df["modelo"].unique())

if MLP not in modelos_disponibles or FISICO not in modelos_disponibles:
    raise ValueError(
        "No se encontraron los dos modelos requeridos "
        "en el archivo de predicciones."
    )

resumenes = []
diagnosticos_segmentos = []

print("\nDIAGNÓSTICO DE ESTABILIDAD DEL SUBSAMPLING")
print("=" * 100)


# ============================================================
# ANÁLISIS POR HORIZONTE
# ============================================================

for horizonte in sorted(BLOQUES_POR_HORIZONTE):

    grupo = df[
        df["horizonte_horas"] == horizonte
    ].copy()

    segmentos = construir_segmentos(grupo)

    if not segmentos:
        print(f"\nHorizonte {horizonte}: no hay segmentos.")
        continue

    longitudes = np.array([
        len(s["mae_mlp"])
        for s in segmentos
    ])

    n_total = int(longitudes.sum())

    metricas_completas = calcular_metricas(segmentos)

    print("\n" + "=" * 100)
    print(f"HORIZONTE: {horizonte} h")
    print("=" * 100)

    print("Observaciones:", n_total)
    print("Segmentos:", len(segmentos))
    print("Longitud mínima:", longitudes.min())
    print("Longitud mediana:", np.median(longitudes))
    print("Longitud máxima:", longitudes.max())

    print(
        "Diferencia MAE con todos los segmentos:",
        f"{metricas_completas['delta_mae']:.4f}"
    )

    print(
        "Diferencia RMSE con todos los segmentos:",
        f"{metricas_completas['delta_rmse']:.4f}"
    )

    # ========================================================
    # ANÁLISIS POR LONGITUD DE BLOQUE
    # ========================================================

    for bloque in BLOQUES_POR_HORIZONTE[horizonte]:

        segmentos_elegibles = [
            s for s in segmentos
            if len(s["mae_mlp"]) >= bloque
        ]

        if not segmentos_elegibles:
            continue

        n_elegible = sum(
            len(s["mae_mlp"])
            for s in segmentos_elegibles
        )

        cobertura = (
            n_elegible / n_total * 100
        )

        observado = calcular_metricas(
            segmentos_elegibles
        )

        analisis_ventanas = analizar_ventanas(
            segmentos_elegibles,
            bloque
        )

        deltas_mae = analisis_ventanas["deltas_mae"]
        deltas_rmse = analisis_ventanas["deltas_rmse"]

        if len(deltas_mae) == 0:
            continue

        # ====================================================
        # INFLUENCIA DE LOS SEGMENTOS
        # ====================================================

        influencia = diagnosticar_influencia(
            segmentos,
            segmentos_elegibles,
            horizonte,
            bloque,
            observado
        )

        diagnosticos_segmentos.extend(influencia)

        elegibles_influencia = [
            r for r in influencia
            if r["elegible"]
        ]

        cambios_mae = np.array([
            abs(r["cambio_delta_mae"])
            for r in elegibles_influencia
            if pd.notna(r["cambio_delta_mae"])
        ])

        cambios_rmse = np.array([
            abs(r["cambio_delta_rmse"])
            for r in elegibles_influencia
            if pd.notna(r["cambio_delta_rmse"])
        ])

        # Ordenar segmentos por influencia absoluta en MAE.
        ranking = sorted(
            elegibles_influencia,
            key=lambda r: abs(r["cambio_delta_mae"])
            if pd.notna(r["cambio_delta_mae"])
            else -1,
            reverse=True
        )

        # ====================================================
        # RESUMEN DE DISTRIBUCIONES
        # ====================================================

        registro = {
            "horizonte": horizonte,
            "bloque_horas": bloque,
            "observaciones_totales": n_total,
            "observaciones_elegibles": n_elegible,
            "cobertura_pct": cobertura,
            "segmentos_totales": len(segmentos),
            "segmentos_elegibles": len(segmentos_elegibles),
            "ventanas_disponibles": len(deltas_mae),

            "delta_mae_todos_segmentos":
                metricas_completas["delta_mae"],

            "delta_rmse_todos_segmentos":
                metricas_completas["delta_rmse"],

            "delta_mae_elegibles":
                observado["delta_mae"],

            "delta_rmse_elegibles":
                observado["delta_rmse"],

            "media_ventanas_mae":
                np.mean(deltas_mae),

            "mediana_ventanas_mae":
                np.median(deltas_mae),

            "media_ventanas_rmse":
                np.mean(deltas_rmse),

            "mediana_ventanas_rmse":
                np.median(deltas_rmse),

            "media_ponderada_mae_superposicion":
                analisis_ventanas["media_ponderada_mae"],

            "max_cambio_absoluto_loo_mae":
                np.max(cambios_mae)
                if len(cambios_mae) else np.nan,

            "mediana_cambio_absoluto_loo_mae":
                np.median(cambios_mae)
                if len(cambios_mae) else np.nan,

            "max_cambio_absoluto_loo_rmse":
                np.max(cambios_rmse)
                if len(cambios_rmse) else np.nan,

            "mediana_cambio_absoluto_loo_rmse":
                np.median(cambios_rmse)
                if len(cambios_rmse) else np.nan,

            "segmento_mas_influyente_mae":
                ranking[0]["id_segmento"]
                if ranking else np.nan,

            "mayor_proporcion_contribucion_mae":
                max(
                    (
                        r["proporcion_contribucion_absoluta_mae"]
                        for r in elegibles_influencia
                        if pd.notna(
                            r["proporcion_contribucion_absoluta_mae"]
                        )
                    ),
                    default=np.nan
                )
        }

        registro.update(
            estadisticas_distribucion(
                deltas_mae,
                "ventanas_mae"
            )
        )

        registro.update(
            estadisticas_distribucion(
                deltas_rmse,
                "ventanas_rmse"
            )
        )

        resumenes.append(registro)

        # ====================================================
        # SALIDA EN TERMINAL
        # ====================================================

        print("\n" + "-" * 100)
        print(f"BLOQUE: {bloque} horas")
        print("-" * 100)

        print(
            f"Cobertura: {cobertura:.2f}% "
            f"({n_elegible}/{n_total} observaciones)"
        )

        print("Segmentos elegibles:", len(segmentos_elegibles))
        print("Ventanas:", len(deltas_mae))

        print("\nDIFERENCIA MAE")
        print(
            f"Global con todos los segmentos: "
            f"{metricas_completas['delta_mae']:.4f}"
        )
        print(
            f"Global con segmentos elegibles: "
            f"{observado['delta_mae']:.4f}"
        )
        print(f"Media de ventanas: {np.mean(deltas_mae):.4f}")
        print(f"Mediana de ventanas: {np.median(deltas_mae):.4f}")
        print(
            "Percentiles 5-95:",
            f"[{np.quantile(deltas_mae, 0.05):.4f}, "
            f"{np.quantile(deltas_mae, 0.95):.4f}]"
        )
        print(
            "Media ponderada por superposición:",
            f"{analisis_ventanas['media_ponderada_mae']:.4f}"
        )

        print("\nDIFERENCIA RMSE")
        print(
            f"Global con todos los segmentos: "
            f"{metricas_completas['delta_rmse']:.4f}"
        )
        print(
            f"Global con segmentos elegibles: "
            f"{observado['delta_rmse']:.4f}"
        )
        print(f"Media de ventanas: {np.mean(deltas_rmse):.4f}")
        print(f"Mediana de ventanas: {np.median(deltas_rmse):.4f}")
        print(
            "Percentiles 5-95:",
            f"[{np.quantile(deltas_rmse, 0.05):.4f}, "
            f"{np.quantile(deltas_rmse, 0.95):.4f}]"
        )

        print("\nINFLUENCIA DE SEGMENTOS")
        print(
            "Mayor cambio absoluto al retirar un segmento (MAE):",
            f"{np.max(cambios_mae):.4f}"
            if len(cambios_mae) else "No disponible"
        )

        print(
            "Mediana del cambio absoluto (MAE):",
            f"{np.median(cambios_mae):.4f}"
            if len(cambios_mae) else "No disponible"
        )

        if ranking:
            print(
                "Segmento más influyente:",
                ranking[0]["id_segmento"]
            )

            print(
                "Cambio MAE al retirarlo:",
                f"{ranking[0]['cambio_delta_mae']:.4f}"
            )


# ============================================================
# EXPORTAR RESULTADOS
# ============================================================

Path(SALIDA_RESUMEN).parent.mkdir(
    parents=True,
    exist_ok=True
)

df_resumen = pd.DataFrame(resumenes)

df_segmentos = pd.DataFrame(
    diagnosticos_segmentos
)

df_resumen.to_csv(
    SALIDA_RESUMEN,
    index=False,
    encoding="utf-8"
)

df_segmentos.to_csv(
    SALIDA_SEGMENTOS,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 100)
print("DIAGNÓSTICO FINALIZADO")
print("=" * 100)

print("Resumen:", SALIDA_RESUMEN)
print("Diagnóstico por segmentos:", SALIDA_SEGMENTOS)

print("Configuraciones analizadas:", len(df_resumen))
print("Registros de segmentos:", len(df_segmentos))
