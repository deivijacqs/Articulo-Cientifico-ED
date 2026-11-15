
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. Configuración y rutas
# ============================================================
RUTA_ENTRADA = Path("outputs/tables/o3_dataset_meteorologia.csv")
CARPETA_SALIDA = Path("outputs/tables")
RUTA_PREDICCIONES = CARPETA_SALIDA / "o3_mlp_predictions.csv"
RUTA_METRICAS = CARPETA_SALIDA / "o3_mlp_metrics.csv"

# Variables meteorológicas disponibles
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
# 2. Cargar y ordenar los datos
# ============================================================
df = pd.read_csv(RUTA_ENTRADA)

# Asegurar el formato temporal UTC y orden cronológico
df["fecha_utc"] = pd.to_datetime(df["fecha_utc"], utc=True)
df = df.sort_values("fecha_utc").reset_index(drop=True)

# Comprobar las columnas necesarias antes de entrenar
columnas_necesarias = [
    "fecha_utc",
    "valor_o3_modelo",
    "observacion_entrenable",
    "particion",
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos",
] + VARIABLES_METEO

faltantes = [c for c in columnas_necesarias if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas necesarias: {faltantes}")

# El objetivo solo se usa cuando la observación está habilitada
df["objetivo"] = pd.to_numeric(df["valor_o3_modelo"], errors="coerce")

# ============================================================
# 3. Crear variables predictoras sin usar información futura
# ============================================================
# El objetivo en la fila t es el ozono de la hora t.
# Los retardos se obtienen de horas anteriores de la cuadrícula.
df["o3_lag_1"] = df["objetivo"].shift(1)
df["o3_lag_24"] = df["objetivo"].shift(24)

# Usamos meteorología de t-1 para predecir el ozono de t.
# Esto evita utilizar valores meteorológicos posteriores al objetivo.
for variable in VARIABLES_METEO:
    df[f"{variable}_lag_1"] = df[variable].shift(1)

# Las características cíclicas representan la hora que se predice.
# Son conocidas de antemano y no dependen del valor futuro del ozono.
VARIABLES_TIEMPO = [
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos",
]

VARIABLES_LAG_METEO = [
    f"{variable}_lag_1" for variable in VARIABLES_METEO
]

VARIABLES_X = (
    ["o3_lag_1", "o3_lag_24"]
    + VARIABLES_LAG_METEO
    + VARIABLES_TIEMPO
)

# ============================================================
# 4. Preparar las filas utilizables
# ============================================================
# Solo se entrena/evalúa cuando:
# - existe un objetivo numérico habilitado;
# - existen todos los retardos necesarios;
# - la partición temporal está identificada.
df["objetivo_valido"] = (
    df["observacion_entrenable"].astype(bool)
    & df["objetivo"].notna()
)

# Eliminar filas sin características completas, sin interpolar el ozono
datos = df.dropna(subset=VARIABLES_X + ["objetivo"]).copy()

# Separar por la partición cronológica predefinida
train = datos[
    (datos["particion"] == "train") & datos["objetivo_valido"]
].copy()

validation = datos[
    (datos["particion"] == "validation") & datos["objetivo_valido"]
].copy()

test = datos[
    (datos["particion"] == "test") & datos["objetivo_valido"]
].copy()

if min(len(train), len(validation), len(test)) == 0:
    raise ValueError("Una partición quedó sin datos utilizables.")

X_train = train[VARIABLES_X]
y_train = train["objetivo"]

X_val = validation[VARIABLES_X]
y_val = validation["objetivo"]

X_test = test[VARIABLES_X]
y_test = test["objetivo"]

print("\n=== TAMAÑO DE LAS PARTICIONES UTILIZABLES ===")
print("Entrenamiento:", len(train))
print("Validación:", len(validation))
print("Prueba:", len(test))

# ============================================================
# 5. Escalar usando únicamente el entrenamiento
# ============================================================
# El escalador de características aprende media/desviación solo
# del entrenamiento. Así se evita filtrar información de validación
# y prueba hacia el proceso de ajuste.
escalador_X = StandardScaler()
X_train_scaled = escalador_X.fit_transform(X_train)
X_val_scaled = escalador_X.transform(X_val)
X_test_scaled = escalador_X.transform(X_test)

# Escalar también el objetivo para facilitar la optimización del MLP.
# El escalador del objetivo se ajusta exclusivamente con entrenamiento.
escalador_y = StandardScaler()
y_train_scaled = escalador_y.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).ravel()

# ============================================================
# 6. Entrenar el perceptrón multicapa
# ============================================================
# Dos capas ocultas; sin early stopping aleatorio, para mantener
# el control temporal de la validación.
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

modelo.fit(X_train_scaled, y_train_scaled)

# ============================================================
# 7. Generar predicciones
# ============================================================
def predecir(X_escalado):
    """Predice en escala original de concentración de ozono."""
    pred_scaled = modelo.predict(X_escalado)
    return escalador_y.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()

pred_val = predecir(X_val_scaled)
pred_test = predecir(X_test_scaled)

# Persistencia: el mejor valor disponible de la hora anterior.
# Se compara sobre las mismas filas que la red neuronal.
persistencia_val = validation["o3_lag_1"].to_numpy()
persistencia_test = test["o3_lag_1"].to_numpy()

# ============================================================
# 8. Métricas con comparaciones pareadas
# ============================================================
def calcular_metricas(y_real, y_pred):
    """Calcula métricas en la escala original del ozono."""
    return {
        "n": len(y_real),
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "R2": r2_score(y_real, y_pred),
    }

resultados = []

for nombre, particion, real, pred_mlp, pred_persistencia in [
    ("validation", validation, y_val.to_numpy(), pred_val, persistencia_val),
    ("test", test, y_test.to_numpy(), pred_test, persistencia_test),
]:
    for metodo, pred in [
        ("MLP_meteorologia", pred_mlp),
        ("Persistencia_lag_1", pred_persistencia),
    ]:
        metricas = calcular_metricas(real, pred)
        resultados.append({
            "particion": nombre,
            "metodo": metodo,
            **metricas,
        })

metricas_df = pd.DataFrame(resultados)

# ============================================================
# 9. Guardar predicciones y métricas
# ============================================================
salidas = []

for nombre, particion, real, pred_mlp, pred_persistencia in [
    ("validation", validation, y_val.to_numpy(), pred_val, persistencia_val),
    ("test", test, y_test.to_numpy(), pred_test, persistencia_test),
]:
    salida = pd.DataFrame({
        "fecha_utc": particion["fecha_utc"].to_numpy(),
        "particion": nombre,
        "ozono_real": real,
        "prediccion_mlp": pred_mlp,
        "prediccion_persistencia": pred_persistencia,
    })
    salidas.append(salida)

predicciones_df = pd.concat(salidas, ignore_index=True)

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
predicciones_df.to_csv(RUTA_PREDICCIONES, index=False)
metricas_df.to_csv(RUTA_METRICAS, index=False)

# ============================================================
# 10. Mostrar el resultado
# ============================================================
print("\n=== MÉTRICAS DEL MODELO ===")
print(metricas_df.round(4).to_string(index=False))

print("\nIteraciones realizadas:", modelo.n_iter_)
print("Convergencia alcanzada:", modelo.n_iter_ < modelo.max_iter)
print("Archivo de métricas:", RUTA_METRICAS)
print("Archivo de predicciones:", RUTA_PREDICCIONES)
