
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. Configuración
# ============================================================
RUTA_ENTRADA = Path("outputs/tables/o3_dataset_meteorologia.csv")
RUTA_SALIDA = Path("outputs/tables/o3_multihorizon_metrics.csv")

HORIZONTES = [1, 6, 24]

VARIABLES_METEO = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_sin",
    "wind_direction_cos",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
]

# ============================================================
# 2. Cargar datos y validar estructura temporal
# ============================================================
df = pd.read_csv(RUTA_ENTRADA)
df["fecha_utc"] = pd.to_datetime(df["fecha_utc"], utc=True)
df = df.sort_values("fecha_utc").reset_index(drop=True)

if df["fecha_utc"].duplicated().any():
    raise ValueError("Hay timestamps duplicados.")

# Comprobar que la serie sea una cuadrícula horaria continua
diferencias = df["fecha_utc"].diff().dropna()
if not diferencias.eq(pd.Timedelta(hours=1)).all():
    raise ValueError("La serie no tiene intervalos horarios regulares.")

df["objetivo"] = pd.to_numeric(df["valor_o3_modelo"], errors="coerce")
df["objetivo_valido"] = (
    df["observacion_entrenable"].astype(bool)
    & df["objetivo"].notna()
)

# Variables conocidas en el instante de predicción:
# concentración actual y concentración de 24 horas antes.
df["o3_actual"] = df["objetivo"]
df["o3_hace_24h"] = df["objetivo"].shift(24)

# Variables cíclicas del instante futuro que se quiere predecir.
# Se conocen de antemano y no contienen el valor futuro del ozono.
VARIABLES_TIEMPO_OBJETIVO = [
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos",
]

# Variables base y meteorológicas
VARIABLES_BASE = [
    "o3_actual",
    "o3_hace_24h",
] + VARIABLES_TIEMPO_OBJETIVO

VARIABLES_COMPLETAS = VARIABLES_BASE + VARIABLES_METEO

# Guardar particiones para comprobar que no se cruzan los límites
particiones = df["particion"].astype(str)

# ============================================================
# 3. Métricas
# ============================================================
def calcular_metricas(real, prediccion):
    return {
        "n": len(real),
        "MAE": mean_absolute_error(real, prediccion),
        "RMSE": np.sqrt(mean_squared_error(real, prediccion)),
        "R2": r2_score(real, prediccion),
    }


def entrenar_mlp(train, X_train, y_train, X_eval):
    # El escalador se ajusta exclusivamente con entrenamiento
    escalador_x = StandardScaler()
    X_train_s = escalador_x.fit_transform(X_train)
    X_eval_s = escalador_x.transform(X_eval)

    # El escalador del objetivo también usa solo entrenamiento
    escalador_y = StandardScaler()
    y_train_s = escalador_y.fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    ).ravel()

    # Misma arquitectura para comparar de forma consistente
    modelo = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=400,
        early_stopping=False,
        random_state=42,
    )

    modelo.fit(X_train_s, y_train_s)

    # Convertir predicciones a unidades originales
    pred_scaled = modelo.predict(X_eval_s)
    pred = escalador_y.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()

    return pred, modelo.n_iter_


# ============================================================
# 4. Evaluar cada horizonte de predicción
# ============================================================
resultados = []

for horizonte in HORIZONTES:
    print(f"\n{'=' * 55}")
    print(f"HORIZONTE: {horizonte} hora(s)")
    print(f"{'=' * 55}")

    # Cada fila representa el instante objetivo.
    # Las entradas se desplazan hacia atrás hasta el origen.
    datos = df.copy()

    datos["o3_actual"] = df["objetivo"].shift(horizonte)
    datos["o3_hace_24h"] = df["objetivo"].shift(horizonte + 24)

    # La meteorología procede del instante de origen, no del futuro.
    for variable in VARIABLES_METEO:
        datos[variable] = df[variable].shift(horizonte)

    # Recuperar las características temporales del instante objetivo
    for variable in VARIABLES_TIEMPO_OBJETIVO:
        datos[variable] = df[variable]

    # Etiqueta: concentración observada en el instante objetivo
    datos["y"] = df["objetivo"]

    # Solo aceptar pares origen-objetivo dentro de la misma partición.
    # Esto evita que un ejemplo de entrenamiento tenga su objetivo
    # dentro de validación o prueba, y viceversa.
    datos["particion_origen"] = particiones.shift(horizonte)

    mascara_origen_valido = (
        df["objetivo_valido"].shift(horizonte).fillna(False).astype(bool)
        & df["objetivo_valido"].shift(horizonte + 24).fillna(False).astype(bool)
    )

    mascara_objetivo_valido = df["objetivo_valido"]

    datos = datos[
        mascara_origen_valido
        & mascara_objetivo_valido
        & (datos["particion_origen"] == datos["particion"])
    ].copy()

    # Eliminar filas con características incompletas
    datos = datos.dropna(
        subset=VARIABLES_COMPLETAS + ["y"]
    )

    train = datos[datos["particion"] == "train"].copy()
    validation = datos[datos["particion"] == "validation"].copy()
    test = datos[datos["particion"] == "test"].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            f"Partición vacía para el horizonte de {horizonte} h."
        )

    print("Filas utilizables:")
    print("  Entrenamiento:", len(train))
    print("  Validación:", len(validation))
    print("  Prueba:", len(test))

    # Entrenar dos modelos sobre las mismas filas de entrenamiento
    for nombre, variables in [
        ("MLP_sin_meteorologia", VARIABLES_BASE),
        ("MLP_con_meteorologia", VARIABLES_COMPLETAS),
    ]:
        X_train = train[variables]
        y_train = train["y"]

        print(f"\nEntrenando: {nombre}")

        for nombre_particion, conjunto in [
            ("validation", validation),
            ("test", test),
        ]:
            X_eval = conjunto[variables]
            y_real = conjunto["y"].to_numpy()

            # Entrenar con train y predecir el conjunto seleccionado
            pred, iteraciones = entrenar_mlp(
                train, X_train, y_train, X_eval
            )

            metricas = calcular_metricas(y_real, pred)
            resultados.append({
                "horizonte_horas": horizonte,
                "modelo": nombre,
                "particion": nombre_particion,
                **metricas,
                "iteraciones": iteraciones,
            })

    # Persistencia: mantener el valor de ozono del instante de origen.
    # Se evalúa sobre las mismas filas que los MLP.
    for nombre_particion, conjunto in [
        ("validation", validation),
        ("test", test),
    ]:
        y_real = conjunto["y"].to_numpy()
        pred_persistencia = conjunto["o3_actual"].to_numpy()

        metricas = calcular_metricas(y_real, pred_persistencia)
        resultados.append({
            "horizonte_horas": horizonte,
            "modelo": "Persistencia",
            "particion": nombre_particion,
            **metricas,
            "iteraciones": 0,
        })

# ============================================================
# 5. Guardar y mostrar resultados
# ============================================================
metricas_df = pd.DataFrame(resultados)

# Ordenar para facilitar la comparación entre horizontes
metricas_df = metricas_df.sort_values(
    ["horizonte_horas", "particion", "modelo"]
).reset_index(drop=True)

RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
metricas_df.to_csv(RUTA_SALIDA, index=False)

print("\n=== RESULTADOS MULTIHORIZONTE ===")
print(metricas_df.round(4).to_string(index=False))
print("\nArchivo guardado:", RUTA_SALIDA)
