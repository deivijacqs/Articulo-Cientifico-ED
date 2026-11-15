
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = "outputs/tables/o3_pinn_predicciones_detalladas.csv"

SALIDA = "outputs/tables/34_block_subsampling_temporal.csv"

MLP = "MLP_con_meteorologia"
FISICO = "Modelo_neuronal_fisico"

# Longitudes de ventana por horizonte.
# Se mantienen las alternativas evaluadas anteriormente.
BLOQUES_POR_HORIZONTE = {
    1: [2, 4, 6],
    6: [2, 4, 6],
    24: [6, 8, 12]
}

# Cuantiles para el intervalo de confianza.
ALFA = 0.05

# Cobertura mínima orientativa para advertir sobre
# configuraciones con pocos datos elegibles.
COBERTURA_MINIMA = 80.0

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def construir_segmentos(grupo):
    """
    Construye segmentos horarios continuos utilizando
    exclusivamente fechas comunes a ambos modelos.

    No se permite que una ventana atraviese interrupciones
    temporales.
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

    # Una diferencia distinta de una hora inicia
    # un nuevo segmento.
    diferencias = (
        matriz.index.to_series()
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    ids = diferencias.ne(1).cumsum()

    segmentos = []

    for _, indices in ids.groupby(ids).groups.items():

        segmento = matriz.loc[indices].sort_index()

        segmentos.append({
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


def calcular_diferencias(segmentos):
    """
    Calcula las diferencias observadas sobre un conjunto
    de segmentos.

    Diferencia positiva:
    favorece al modelo neuronal físico.
    """

    total = sum(
        len(s["mae_mlp"])
        for s in segmentos
    )

    if total == 0:
        return None

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

    mae_mlp = suma_mae_mlp / total
    mae_fisico = suma_mae_fisico / total

    rmse_mlp = np.sqrt(
        suma_mse_mlp / total
    )

    rmse_fisico = np.sqrt(
        suma_mse_fisico / total
    )

    return {
        "n": total,
        "mae_mlp": mae_mlp,
        "mae_fisico": mae_fisico,
        "delta_mae": mae_mlp - mae_fisico,
        "rmse_mlp": rmse_mlp,
        "rmse_fisico": rmse_fisico,
        "delta_rmse": rmse_mlp - rmse_fisico
    }


def extraer_ventanas(segmentos, longitud):
    """
    Extrae todas las ventanas móviles consecutivas
    de longitud fija.

    Las ventanas nunca atraviesan huecos temporales.

    Cada ventana contiene exactamente 'longitud'
    observaciones horarias.
    """

    ventanas = []

    for indice_segmento, segmento in enumerate(segmentos):

        n = len(segmento["mae_mlp"])

        if n < longitud:
            continue

        for inicio in range(n - longitud + 1):

            fin = inicio + longitud

            ventana = {
                clave: valores[inicio:fin]
                for clave, valores in segmento.items()
            }

            ventanas.append(ventana)

    return ventanas


def calcular_diferencia_ventana(ventana):
    """
    Calcula las diferencias de MAE y RMSE
    dentro de una ventana individual.
    """

    mae_mlp = np.mean(ventana["mae_mlp"])
    mae_fisico = np.mean(ventana["mae_fisico"])

    rmse_mlp = np.sqrt(
        np.mean(ventana["mse_mlp"])
    )

    rmse_fisico = np.sqrt(
        np.mean(ventana["mse_fisico"])
    )

    return {
        "delta_mae": mae_mlp - mae_fisico,
        "delta_rmse": rmse_mlp - rmse_fisico
    }


def calcular_intervalo_subsampling(
    valores_bloque,
    estimador_observado,
    longitud_bloque,
    n_observaciones
):
    """
    Calcula un intervalo de subsampling.

    Estadístico de subsampling:

        Z_b = sqrt(b) * (theta_b - theta_n)

    Intervalo aproximado:

        [theta_n - q_97.5 / sqrt(n),
         theta_n - q_2.5 / sqrt(n)]

    donde:
        b = longitud del bloque
        n = observaciones elegibles
        theta_b = estimador de cada ventana
        theta_n = estimador de todas las observaciones elegibles

    La interpretación inferencial requiere condiciones
    de estacionariedad y dependencia débil que deben
    evaluarse por separado.
    """

    valores_bloque = np.asarray(
        valores_bloque,
        dtype=float
    )

    # Centrar los estimadores de las ventanas
    # respecto al estimador observado.
    z = np.sqrt(longitud_bloque) * (
        valores_bloque - estimador_observado
    )

    q_inferior = np.quantile(
        z,
        ALFA / 2
    )

    q_superior = np.quantile(
        z,
        1 - ALFA / 2
    )

    escala = np.sqrt(n_observaciones)

    limite_inferior = (
        estimador_observado
        - q_superior / escala
    )

    limite_superior = (
        estimador_observado
        - q_inferior / escala
    )

    return {
        "ic95_inferior": limite_inferior,
        "ic95_superior": limite_superior,
        "cuantil_z_025": q_inferior,
        "cuantil_z_975": q_superior,
        "desviacion_z": np.std(z, ddof=1)
    }


# ============================================================
# CARGA DE DATOS
# ============================================================

df = pd.read_csv(ARCHIVO)

df["fecha_objetivo"] = pd.to_datetime(
    df["fecha_objetivo"],
    utc=True
)

# Solo se utiliza validación.
df = df[
    df["particion"] == "validation"
].copy()

resultados = []

print("\nBLOCK SUBSAMPLING TEMPORAL")
print("=" * 95)

# ============================================================
# ANÁLISIS POR HORIZONTE
# ============================================================

for horizonte, grupo in df.groupby("horizonte_horas"):

    segmentos = construir_segmentos(grupo)

    if not segmentos:
        continue

    longitudes = np.array([
        len(s["mae_mlp"])
        for s in segmentos
    ])

    n_total = int(longitudes.sum())

    print(f"\nHORIZONTE: {horizonte} h")
    print("-" * 95)
    print("Observaciones comunes:", n_total)
    print("Segmentos:", len(segmentos))
    print("Longitud mediana:", np.median(longitudes))
    print("Longitud máxima:", longitudes.max())

    # ========================================================
    # ANÁLISIS POR LONGITUD DE BLOQUE
    # ========================================================

    for bloque in BLOQUES_POR_HORIZONTE[horizonte]:

        # Excluir únicamente los segmentos que no pueden
        # contener una ventana completa de esta longitud.
        segmentos_elegibles = [
            s for s in segmentos
            if len(s["mae_mlp"]) >= bloque
        ]

        if not segmentos_elegibles:
            print(f"\nBloque {bloque} h: sin segmentos elegibles.")
            continue

        n_elegible = sum(
            len(s["mae_mlp"])
            for s in segmentos_elegibles
        )

        cobertura = (
            n_elegible / n_total * 100
        )

        # Estimador observado sobre la misma población
        # que se utilizará para extraer las ventanas.
        observado = calcular_diferencias(
            segmentos_elegibles
        )

        # Extraer todas las ventanas posibles.
        ventanas = extraer_ventanas(
            segmentos_elegibles,
            bloque
        )

        if len(ventanas) < 30:
            print(
                f"\nBloque {bloque} h: "
                f"solo {len(ventanas)} ventanas. "
                "Cobertura insuficiente para el análisis."
            )
            continue

        # Calcular las diferencias de cada ventana.
        deltas_mae = []
        deltas_rmse = []

        for ventana in ventanas:

            diferencia = calcular_diferencia_ventana(
                ventana
            )

            deltas_mae.append(
                diferencia["delta_mae"]
            )

            deltas_rmse.append(
                diferencia["delta_rmse"]
            )

        # ====================================================
        # INTERVALOS DE SUBSAMPLING
        # ====================================================

        intervalo_mae = calcular_intervalo_subsampling(
            deltas_mae,
            observado["delta_mae"],
            bloque,
            n_elegible
        )

        intervalo_rmse = calcular_intervalo_subsampling(
            deltas_rmse,
            observado["delta_rmse"],
            bloque,
            n_elegible
        )

        # ====================================================
        # DIAGNÓSTICOS DE LAS VENTANAS
        # ====================================================

        media_ventanas_mae = np.mean(deltas_mae)
        media_ventanas_rmse = np.mean(deltas_rmse)

        mediana_ventanas_mae = np.median(deltas_mae)
        mediana_ventanas_rmse = np.median(deltas_rmse)

        print(f"\nBloque: {bloque} horas")
        print("Segmentos elegibles:", len(segmentos_elegibles))
        print("Observaciones elegibles:", n_elegible)
        print(f"Cobertura: {cobertura:.2f} %")
        print("Ventanas disponibles:", len(ventanas))

        if cobertura < COBERTURA_MINIMA:
            print(
                "ADVERTENCIA: cobertura inferior al "
                f"{COBERTURA_MINIMA:.0f} %."
            )

        print("\nDIFERENCIA DE MAE")
        print(
            f"Observada: {observado['delta_mae']:.4f}"
        )
        print(
            f"Media de ventanas: {media_ventanas_mae:.4f}"
        )
        print(
            f"Mediana de ventanas: {mediana_ventanas_mae:.4f}"
        )
        print(
            "Intervalo subsampling 95 %: "
            f"[{intervalo_mae['ic95_inferior']:.4f}, "
            f"{intervalo_mae['ic95_superior']:.4f}]"
        )

        print("\nDIFERENCIA DE RMSE")
        print(
            f"Observada: {observado['delta_rmse']:.4f}"
        )
        print(
            f"Media de ventanas: {media_ventanas_rmse:.4f}"
        )
        print(
            f"Mediana de ventanas: {mediana_ventanas_rmse:.4f}"
        )
        print(
            "Intervalo subsampling 95 %: "
            f"[{intervalo_rmse['ic95_inferior']:.4f}, "
            f"{intervalo_rmse['ic95_superior']:.4f}]"
        )

        # ====================================================
        # GUARDAR RESULTADOS
        # ====================================================

        for metrica, observado_delta, media_ventanas, mediana_ventanas, intervalo in [
            (
                "delta_mae",
                observado["delta_mae"],
                media_ventanas_mae,
                mediana_ventanas_mae,
                intervalo_mae
            ),
            (
                "delta_rmse",
                observado["delta_rmse"],
                media_ventanas_rmse,
                mediana_ventanas_rmse,
                intervalo_rmse
            )
        ]:

            resultados.append({
                "horizonte": horizonte,
                "bloque_horas": bloque,
                "metrica": metrica,
                "observaciones_totales": n_total,
                "observaciones_elegibles": n_elegible,
                "cobertura_pct": cobertura,
                "segmentos_totales": len(segmentos),
                "segmentos_elegibles": len(segmentos_elegibles),
                "ventanas_disponibles": len(ventanas),
                "diferencia_observada": observado_delta,
                "media_ventanas": media_ventanas,
                "mediana_ventanas": mediana_ventanas,
                **intervalo
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

print("\n" + "=" * 95)
print("BLOCK SUBSAMPLING FINALIZADO")
print("Archivo:", SALIDA)
print("Resultados guardados:", len(salida))
