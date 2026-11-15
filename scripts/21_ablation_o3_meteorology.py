
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. Cargar los datos ya integrados
# ============================================================
RUTA = Path("outputs/tables/o3_dataset_meteorologia.csv")
SALIDA = Path("outputs/tables/o3_ablation_metrics.csv")

df = pd.read_csv(RUTA)
df["fecha_utc"] = pd.to_datetime(df["fecha_utc"], utc=True)
df = df.sort_values("fecha_utc").reset_index(drop=True)

# El objetivo conserva la máscara de calidad definida previamente.
df["objetivo"] = pd.to_numeric(df["valor_o3_modelo"], errors="coerce")

# ============================================================
# 2. Construir retardos causales
# ============================================================
# Solo usamos valores de ozono de horas anteriores.
df["o3_lag_1"] = df["objetivo"].shift(1)
df["o3_lag_24"] = df["objetivo"].shift(24)

meteo = [
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

# Meteorología de la hora anterior, no posterior al objetivo.
for variable in meteo:
    df[f"{variable}_lag_1"] = df[variable].shift(1)

# Modelo reducido: retardos de ozono y variables temporales.
variables_base = [
    "o3_lag_1",
    "o3_lag_24",
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos",
]

# Modelo completo: mismas variables más meteorología.
variables_completas = variables_base + [
    f"{variable}_lag_1" for variable in meteo
]

# ============================================================
# 3. Usar exactamente las mismas filas para ambos modelos
# ============================================================
# Esto garantiza que la comparación sea pareada y justa.
df["objetivo_valido"] = (
    df["observacion_entrenable"].astype(bool)
    & df["objetivo"].notna()
)

todas_las_variables = variables_completas + ["objetivo"]
datos = df.dropna(subset=todas_las_variables).copy()
datos = datos[datos["objetivo_valido"]].copy()

train = datos[datos["particion"] == "train"].copy()
val = datos[datos["particion"] == "validation"].copy()
test = datos[datos["particion"] == "test"].copy()

if min(len(train), len(val), len(test)) == 0:
    raise ValueError("Una partición no contiene observaciones utilizables.")

print("Filas comunes por partición:")
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

# ============================================================
# 4. Función de entrenamiento y evaluación
# ============================================================
def evaluar_modelo(nombre, variables):
    # Separar las características de cada partición
    X_train = train[variables]
    X_val = val[variables]
    X_test = test[variables]

    y_train = train["objetivo"].to_numpy()
    y_val = val["objetivo"].to_numpy()
    y_test = test["objetivo"].to_numpy()

    # Ajustar el escalador solo con entrenamiento
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_val_s = scaler_x.transform(X_val)
    X_test_s = scaler_x.transform(X_test)

    # Escalar el objetivo con datos de entrenamiento solamente
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(
        y_train.reshape(-1, 1)
    ).ravel()

    # Mantener la misma arquitectura y configuración en ambos modelos
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

    # Volver a las unidades originales de ozono
    pred_val = scaler_y.inverse_transform(
        modelo.predict(X_val_s).reshape(-1, 1)
    ).ravel()

    pred_test = scaler_y.inverse_transform(
        modelo.predict(X_test_s).reshape(-1, 1)
    ).ravel()

    resultados = []

    for particion, real, pred in [
        ("validation", y_val, pred_val),
        ("test", y_test, pred_test),
    ]:
        resultados.append({
            "modelo": nombre,
            "particion": particion,
            "n": len(real),
            "MAE": mean_absolute_error(real, pred),
            "RMSE": np.sqrt(mean_squared_error(real, pred)),
            "R2": r2_score(real, pred),
            "iteraciones": modelo.n_iter_,
        })

    return resultados

# ============================================================
# 5. Ejecutar la ablación
# ============================================================
resultados = []

resultados.extend(
    evaluar_modelo("MLP_sin_meteorologia", variables_base)
)

resultados.extend(
    evaluar_modelo("MLP_con_meteorologia", variables_completas)
)

metricas = pd.DataFrame(resultados)

# Calcular la mejora porcentual frente al modelo sin meteorología.
# Una cifra positiva significa que el error disminuyó.
for particion in ["validation", "test"]:
    base = metricas[
        (metricas["modelo"] == "MLP_sin_meteorologia")
        & (metricas["particion"] == particion)
    ].iloc[0]

    completo = metricas[
        (metricas["modelo"] == "MLP_con_meteorologia")
        & (metricas["particion"] == particion)
    ].iloc[0]

    print(f"\n=== Comparación en {particion} ===")
    print(f"MAE sin meteorología: {base['MAE']:.4f}")
    print(f"MAE con meteorología: {completo['MAE']:.4f}")
    print(
        "Reducción MAE (%):",
        f"{100 * (base['MAE'] - completo['MAE']) / base['MAE']:.2f}"
    )
    print(f"RMSE sin meteorología: {base['RMSE']:.4f}")
    print(f"RMSE con meteorología: {completo['RMSE']:.4f}")
    print(
        "Reducción RMSE (%):",
        f"{100 * (base['RMSE'] - completo['RMSE']) / base['RMSE']:.2f}"
    )

SALIDA.parent.mkdir(parents=True, exist_ok=True)
metricas.to_csv(SALIDA, index=False)

print("\n=== MÉTRICAS COMPLETAS ===")
print(metricas.round(4).to_string(index=False))
print("\nArchivo guardado:", SALIDA)
