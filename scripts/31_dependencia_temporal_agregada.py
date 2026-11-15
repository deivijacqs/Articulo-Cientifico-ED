
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = "outputs/tables/o3_pinn_predicciones_detalladas.csv"

SALIDA = "outputs/tables/31_dependencia_temporal_agregada.csv"

MLP = "MLP_con_meteorologia"
FISICO = "Modelo_neuronal_fisico"

# Analizar hasta 168 horas (7 días).
MAX_LAG = 168

# Cobertura mínima para considerar fiable el diagnóstico
# de cada retardo dentro de la secuencia positiva inicial.
MIN_PARES = 200
MIN_SEGMENTOS = 20

# ============================================================
# CARGA DE DATOS
# ============================================================

df = pd.read_csv(ARCHIVO)

df["fecha_objetivo"] = pd.to_datetime(
    df["fecha_objetivo"],
    utc=True
)

# Utilizar exclusivamente validación.
df = df[df["particion"] == "validation"].copy()

# Métricas que vamos a estudiar.
metricas = {
    "error_absoluto": "MAE",
    "error_cuadratico": "MSE"
}

resultados_globales = []

print("\nDEPENDENCIA TEMPORAL AGREGADA POR SEGMENTOS")
print("=" * 85)

# ============================================================
# ANÁLISIS POR HORIZONTE
# ============================================================

for horizonte, grupo in df.groupby("horizonte_horas"):

    # Organizar los errores de ambos modelos.
    matriz = grupo.pivot(
        index="fecha_objetivo",
        columns="modelo",
        values=list(metricas.keys())
    )

    columnas = [
        (metrica, modelo)
        for metrica in metricas
        for modelo in [MLP, FISICO]
    ]

    # Conservar fechas comunes a ambos modelos y métricas.
    matriz = matriz[columnas].dropna().sort_index()

    # Identificar tramos horarios continuos.
    diferencias_horarias = (
        matriz.index.to_series()
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    # Cada interrupción inicia un nuevo segmento.
    ids_segmentos = diferencias_horarias.ne(1).cumsum()

    segmentos = []

    # ========================================================
    # CONSTRUIR DIFERENCIAS DE PÉRDIDA POR SEGMENTO
    # ========================================================

    for _, indices in ids_segmentos.groupby(ids_segmentos).groups.items():

        segmento = matriz.loc[indices].sort_index()

        datos_segmento = {}

        for columna, nombre in metricas.items():

            # Diferencia emparejada de pérdidas:
            # positiva = menor error del modelo físico.
            diferencia = (
                segmento[(columna, MLP)]
                - segmento[(columna, FISICO)]
            )

            # Centrar dentro del segmento para reducir
            # las diferencias de nivel entre episodios.
            datos_segmento[nombre] = (
                diferencia.to_numpy(dtype=float)
                - diferencia.mean()
            )

        segmentos.append(datos_segmento)

    numero_segmentos = len(segmentos)

    print(f"\nHORIZONTE: {horizonte} h")
    print("-" * 85)
    print("Observaciones comunes:", len(matriz))
    print("Segmentos continuos:", numero_segmentos)

    # ========================================================
    # AUTOCOVARIANZA Y AUTOCORRELACIÓN AGREGADAS
    # ========================================================

    for nombre in metricas.values():

        # Denominador común: suma de cuadrados centrados
        # de todas las observaciones de todos los segmentos.
        gamma_cero = sum(
            np.dot(s[nombre], s[nombre])
            for s in segmentos
        )

        resultados_lags = []

        for lag in range(1, MAX_LAG + 1):

            suma_productos = 0.0
            total_pares = 0
            segmentos_con_pares = 0

            # Calcular productos únicamente dentro de cada segmento.
            for segmento in segmentos:

                valores = segmento[nombre]

                if len(valores) <= lag:
                    continue

                # Productos entre observaciones separadas
                # exactamente por lag horas.
                x = valores[:-lag]
                y = valores[lag:]

                suma_productos += np.dot(x, y)

                total_pares += len(x)
                segmentos_con_pares += 1

            if gamma_cero <= 0 or total_pares == 0:
                correlacion = np.nan
            else:
                # Autocorrelación agregada con denominador
                # común para todos los retardos.
                correlacion = suma_productos / gamma_cero

            resultados_lags.append({
                "horizonte": horizonte,
                "metrica": nombre,
                "lag": lag,
                "autocorrelacion": correlacion,
                "pares": total_pares,
                "segmentos": segmentos_con_pares
            })

        # ====================================================
        # SECUENCIA POSITIVA INICIAL
        # ====================================================

        # Agrupar las autocorrelaciones en pares:
        # (rho1 + rho2), (rho3 + rho4), etc.
        # Detenerse cuando un par no sea positivo o no tenga
        # suficiente cobertura temporal.
        suma_positiva = 0.0
        ultimo_lag_valido = 0
        motivo_detencion = "No se alcanzó el criterio de detención"

        for inicio in range(1, MAX_LAG, 2):

            lag_a = resultados_lags[inicio - 1]
            lag_b = resultados_lags[inicio]

            # Exigir cobertura suficiente en ambos retardos.
            cobertura_valida = (
                lag_a["pares"] >= MIN_PARES
                and lag_b["pares"] >= MIN_PARES
                and lag_a["segmentos"] >= MIN_SEGMENTOS
                and lag_b["segmentos"] >= MIN_SEGMENTOS
            )

            if not cobertura_valida:
                motivo_detencion = (
                    f"Cobertura insuficiente desde lag {inicio}"
                )
                break

            suma_par = (
                lag_a["autocorrelacion"]
                + lag_b["autocorrelacion"]
            )

            if not np.isfinite(suma_par) or suma_par <= 0:
                motivo_detencion = (
                    f"Primer par no positivo: lag {inicio}-{inicio + 1}"
                )
                break

            suma_positiva += suma_par
            ultimo_lag_valido = inicio + 1

        # Tiempo integrado de autocorrelación.
        # Es un diagnóstico descriptivo, no un intervalo de confianza.
        tau_integrado = 1 + 2 * suma_positiva

        print(f"\nMÉTRICA: {nombre}")
        print("Retardo | Autocorrelación | Pares | Segmentos")

        # Mostrar retardos seleccionados para mantener
        # una salida de terminal compacta.
        retardos_mostrar = (
            list(range(1, 25))
            + [30, 36, 42, 48, 60, 72, 96, 120, 144, 168]
        )

        for fila in resultados_lags:

            if fila["lag"] not in retardos_mostrar:
                continue

            if fila["pares"] == 0:
                continue

            print(
                f"{fila['lag']:7d} | "
                f"{fila['autocorrelacion']:15.4f} | "
                f"{fila['pares']:5d} | "
                f"{fila['segmentos']:9d}"
            )

        print("\nSecuencia positiva inicial:")
        print("Último retardo incluido:", ultimo_lag_valido)
        print("Tiempo integrado estimado:", round(tau_integrado, 4))
        print("Criterio de detención:", motivo_detencion)

        # Guardar todos los retardos, no solamente los impresos.
        resultados_globales.extend(resultados_lags)

# ============================================================
# GUARDAR RESULTADOS
# ============================================================

salida = pd.DataFrame(resultados_globales)

Path(SALIDA).parent.mkdir(
    parents=True,
    exist_ok=True
)

salida.to_csv(
    SALIDA,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 85)
print("ANÁLISIS FINALIZADO")
print("Resultados guardados en:", SALIDA)
print("Total de estimaciones:", len(salida))
