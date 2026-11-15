
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carga la serie completa para mantener el historial temporal.
ruta = Path("outputs/tables/o3_dataset_particionado.csv")
df = pd.read_csv(ruta, parse_dates=["fecha_utc"])
df = df.sort_values("fecha_utc").reset_index(drop=True)

# Convierte las concentraciones a valores numéricos.
df["o3"] = pd.to_numeric(df["valor_o3_modelo"], errors="coerce")

# Genera predicciones usando únicamente observaciones pasadas.
df["pred_persistencia"] = df["o3"].shift(1)
df["pred_diaria"] = df["o3"].shift(24)

resultados = []

for particion in ["validation", "test"]:
    parte = df[df["particion"] == particion].copy()

    # Conserva solo timestamps donde el objetivo y ambos predictores existen.
    comun = parte[
        ["fecha_utc", "o3", "pred_persistencia", "pred_diaria"]
    ].dropna()

    # Calcula ambas métricas sobre exactamente las mismas horas.
    real = comun["o3"].to_numpy()

    for metodo, columna in [
        ("Persistencia", "pred_persistencia"),
        ("Persistencia diaria", "pred_diaria"),
    ]:
        predicho = comun[columna].to_numpy()

        resultados.append({
            "particion": particion,
            "metodo": metodo,
            "n_timestamps_comunes": len(comun),
            "MAE_ug_m3": mean_absolute_error(real, predicho),
            "RMSE_ug_m3": np.sqrt(mean_squared_error(real, predicho)),
            "R2": r2_score(real, predicho),
        })

# Muestra la comparación justa y guarda los resultados.
tabla = pd.DataFrame(resultados)

print("\n=== COMPARACIÓN SOBRE TIMESTAMPS COMUNES ===")
print(tabla.round(4).to_string(index=False))

salida = Path("outputs/tables/o3_baseline_metrics_paired.csv")
salida.parent.mkdir(parents=True, exist_ok=True)
tabla.to_csv(salida, index=False)

print(f"\nResultados guardados en: {salida}")
