
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

ARCHIVO = "outputs/tables/o3_pinn_predicciones_detalladas.csv"

SALIDA = "outputs/tables/33_bootstrap_jerarquico_bloques.csv"

MLP = "MLP_con_meteorologia"
FISICO = "Modelo_neuronal_fisico"

# Número de réplicas bootstrap.
N_BOOTSTRAP = 2000

# Semilla para reproducibilidad.
SEMILLA = 42

# Longitudes candidatas por horizonte.
BLOQUES_POR_HORIZONTE = {
    1: [2, 4, 6],
    6: [2, 4, 6],
    24: [6, 8, 12]
}

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def construir_segmentos(grupo):
    """
    Construye segmentos horarios continuos.

    Cada segmento contiene las observaciones comunes de ambos
    modelos. No se permite que un bloque atraviese un hueco
    temporal.
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

    # Identificar interrupciones de continuidad horaria.
    diferencias = (
        matriz.index.to_series()
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    ids_segmentos = diferencias.ne(1).cumsum()

    segmentos = []

    for _, indices in ids_segmentos.groupby(ids_segmentos).groups.items():

        segmento = matriz.loc[indices].sort_index()

        # Guardar las pérdidas emparejadas.
        # Los mismos índices se utilizarán para ambos modelos.
        datos = {
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
        }

        segmentos.append(datos)

    return segmentos


def remuestrear_indices(longitud, bloque, rng):
    """
    Genera índices mediante Moving Block Bootstrap no circular.

    Los bloques se extraen únicamente dentro del segmento.
    No se conectan artificialmente el final y el principio
    de un segmento.

    Si el segmento es más corto o igual que el bloque,
    se conserva completo. La variabilidad de ese segmento
    procede del remuestreo externo de segmentos.
    """

    # Segmentos cortos: conservar su secuencia completa.
    if longitud <= bloque:
        return np.arange(longitud)

    indices = []

    while len(indices) < longitud:

        # Inicio aleatorio de un bloque completo.
        inicio = rng.integers(
            0,
            longitud - bloque + 1
        )

        bloque_indices = np.arange(
            inicio,
            inicio + bloque
        )

        indices.extend(bloque_indices.tolist())

    return np.asarray(
        indices[:longitud],
        dtype=int
    )


def calcular_metricas(segmentos, indices_segmentos, bloque, rng):
    """
    Calcula MAE y RMSE para ambos modelos y sus diferencias.

    Diferencia positiva = menor error del modelo físico.

    Los errores de ambos modelos siempre se remuestrean
    con los mismos índices.
    """

    suma_mae_mlp = 0.0
    suma_mae_fisico = 0.0

    suma_mse_mlp = 0.0
    suma_mse_fisico = 0.0

    total_observaciones = 0

    for indice in indices_segmentos:

        segmento = segmentos[indice]

        longitud = len(segmento["mae_mlp"])

        indices = remuestrear_indices(
            longitud,
            bloque,
            rng
        )

        # Aplicar exactamente los mismos índices a ambos modelos.
        suma_mae_mlp += segmento["mae_mlp"][indices].sum()
        suma_mae_fisico += segmento["mae_fisico"][indices].sum()

        suma_mse_mlp += segmento["mse_mlp"][indices].sum()
        suma_mse_fisico += segmento["mse_fisico"][indices].sum()

        total_observaciones += len(indices)

    mae_mlp = suma_mae_mlp / total_observaciones
    mae_fisico = suma_mae_fisico / total_observaciones

    rmse_mlp = np.sqrt(
        suma_mse_mlp / total_observaciones
    )

    rmse_fisico = np.sqrt(
        suma_mse_fisico / total_observaciones
    )

    return {
        "mae_mlp": mae_mlp,
        "mae_fisico": mae_fisico,
        "delta_mae": mae_mlp - mae_fisico,

        "rmse_mlp": rmse_mlp,
        "rmse_fisico": rmse_fisico,
        "delta_rmse": rmse_mlp - rmse_fisico,

        "observaciones_replica": total_observaciones
    }


def resumir_distribucion(valores, observado):
    """
    Resume una distribución bootstrap mediante:
    media, desviación estándar e intervalo percentil del 95 %.
    """

    valores = np.asarray(valores, dtype=float)

    return {
        "observado": observado,
        "media_bootstrap": np.mean(valores),
        "desviacion_bootstrap": np.std(
            valores,
            ddof=1
        ),
        "ic95_inferior": np.percentile(valores, 2.5),
        "ic95_superior": np.percentile(valores, 97.5),

        # Proporción de réplicas donde la diferencia
        # favorece al modelo físico.
        "proporcion_delta_positiva": np.mean(valores > 0)
    }


# ============================================================
# CARGA Y PREPARACIÓN
# ============================================================

df = pd.read_csv(ARCHIVO)

df["fecha_objetivo"] = pd.to_datetime(
    df["fecha_objetivo"],
    utc=True
)

# Utilizar exclusivamente validación.
df = df[
    df["particion"] == "validation"
].copy()

resultados = []

print("\nBOOTSTRAP JERÁRQUICO DE BLOQUES TEMPORALES")
print("=" * 90)

# ============================================================
# ANÁLISIS POR HORIZONTE
# ============================================================

for horizonte, grupo in df.groupby("horizonte_horas"):

    segmentos = construir_segmentos(grupo)

    if not segmentos:
        continue

    numero_segmentos = len(segmentos)

    longitudes = np.array([
        len(segmento["mae_mlp"])
        for segmento in segmentos
    ])

    total_observaciones = int(longitudes.sum())

    # Métricas observadas sobre todas las observaciones comunes.
    suma_mae_mlp = sum(
        s["mae_mlp"].sum()
        for s in segmentos
    )

    suma_mae_fisico = sum(
        s["mae_fisico"].sum()
        for s in segmentos
    )

    suma_mse_mlp = sum(
        s["mse_mlp"].sum()
        for s in segmentos
    )

    suma_mse_fisico = sum(
        s["mse_fisico"].sum()
        for s in segmentos
    )

    observado = {
        "mae_mlp": suma_mae_mlp / total_observaciones,
        "mae_fisico": suma_mae_fisico / total_observaciones,

        "rmse_mlp": np.sqrt(
            suma_mse_mlp / total_observaciones
        ),

        "rmse_fisico": np.sqrt(
            suma_mse_fisico / total_observaciones
        )
    }

    observado["delta_mae"] = (
        observado["mae_mlp"]
        - observado["mae_fisico"]
    )

    observado["delta_rmse"] = (
        observado["rmse_mlp"]
        - observado["rmse_fisico"]
    )

    print(f"\nHORIZONTE: {horizonte} h")
    print("-" * 90)

    print("Observaciones:", total_observaciones)
    print("Segmentos:", numero_segmentos)
    print("Longitud mediana:", np.median(longitudes))
    print("Longitud máxima:", longitudes.max())

    # ========================================================
    # EVALUACIÓN DE CADA LONGITUD DE BLOQUE
    # ========================================================

    for bloque in BLOQUES_POR_HORIZONTE[horizonte]:

        # Segmentos que no permiten extraer un bloque completo.
        cortos = longitudes <= bloque

        numero_cortos = int(cortos.sum())

        observaciones_cortas = int(
            longitudes[cortos].sum()
        )

        cobertura_corta = (
            observaciones_cortas
            / total_observaciones
            * 100
        )

        rng = np.random.default_rng(
            SEMILLA + horizonte * 100 + bloque
        )

        distribuciones = {
            clave: []
            for clave in [
                "mae_mlp",
                "mae_fisico",
                "delta_mae",
                "rmse_mlp",
                "rmse_fisico",
                "delta_rmse"
            ]
        }

        tamanos_replicas = []

        # ====================================================
        # RÉPLICAS BOOTSTRAP
        # ====================================================

        for _ in range(N_BOOTSTRAP):

            # Primer nivel:
            # seleccionar segmentos completos con reemplazo.
            indices_segmentos = rng.integers(
                0,
                numero_segmentos,
                size=numero_segmentos
            )

            # Segundo nivel:
            # remuestrear bloques dentro de cada segmento.
            metricas_replica = calcular_metricas(
                segmentos,
                indices_segmentos,
                bloque,
                rng
            )

            for clave in distribuciones:
                distribuciones[clave].append(
                    metricas_replica[clave]
                )

            tamanos_replicas.append(
                metricas_replica["observaciones_replica"]
            )

        # ====================================================
        # RESUMEN DE RESULTADOS
        # ====================================================

        print(f"\nBloque: {bloque} h")
        print(
            f"Segmentos cortos: {numero_cortos}/{numero_segmentos} "
            f"({numero_cortos / numero_segmentos * 100:.2f} %)"
        )

        print(
            f"Observaciones en segmentos cortos: "
            f"{observaciones_cortas} "
            f"({cobertura_corta:.2f} %)"
        )

        print(
            "Tamaño medio de réplica:",
            round(np.mean(tamanos_replicas), 1)
        )

        print(
            "Tamaño mínimo-máximo:",
            min(tamanos_replicas),
            "-",
            max(tamanos_replicas)
        )

        for clave in distribuciones:

            resumen = resumir_distribucion(
                distribuciones[clave],
                observado[clave]
            )

            resultados.append({
                "horizonte": horizonte,
                "bloque_horas": bloque,
                "metrica": clave,
                "observaciones": total_observaciones,
                "segmentos": numero_segmentos,
                "segmentos_cortos": numero_cortos,
                "observaciones_segmentos_cortos": observaciones_cortas,
                "cobertura_segmentos_cortos_pct": cobertura_corta,
                "replicas": N_BOOTSTRAP,
                "tamano_medio_replica": np.mean(tamanos_replicas),
                **resumen
            })

            # Mostrar únicamente las diferencias entre modelos.
            if clave in ["delta_mae", "delta_rmse"]:

                print(
                    f"{clave}: "
                    f"observado={resumen['observado']:.4f} | "
                    f"media={resumen['media_bootstrap']:.4f} | "
                    f"IC95%=[{resumen['ic95_inferior']:.4f}, "
                    f"{resumen['ic95_superior']:.4f}]"
                )

# ============================================================
# GUARDAR RESULTADOS
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

print("\n" + "=" * 90)
print("BOOTSTRAP FINALIZADO")
print("Archivo:", SALIDA)
print("Comparaciones y métricas guardadas:", len(salida))
print("Réplicas por configuración:", N_BOOTSTRAP)
