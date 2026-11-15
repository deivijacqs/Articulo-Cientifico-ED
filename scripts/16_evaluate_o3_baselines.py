
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carga la serie completa para conservar el historial entre particiones.
RUTA = Path("outputs/tables/o3_dataset_particionado.csv")
df = pd.read_csv(RUTA, parse_dates=["fecha_utc"])
df = df.sort_values("fecha_utc").reset_index(drop=True)

# Convierte la concentración de entrenamiento a valores numéricos.
# Los valores excluidos provisionalmente ya deben aparecer como NaN.
df["o3"] = pd.to_numeric(df["valor_o3_modelo"], errors="coerce")

# Comprueba que las fechas estén ordenadas y separadas por una hora.
if df["fecha_utc"].duplicated().any():
    raise ValueError("Hay fechas duplicadas en la serie.")

if not (df["fecha_utc"].diff().dropna() == pd.Timedelta(hours=1)).all():
    raise ValueError("La serie no mantiene intervalos horarios regulares.")

# Construye predicciones usando exclusivamente observaciones anteriores.
# shift(1) representa la concentración de la hora previa.
df["pred_persistencia"] = df["o3"].shift(1)

# shift(24) representa la concentración de la misma hora del día anterior.
df["pred_diaria"] = df["o3"].shift(24)

# Evalúa cada método en validación y prueba, sin ajustar parámetros con esos datos.
resultados = []

for particion in ["validation", "test"]:
    parte = df[df["particion"] == particion].copy()

    # Solo se evalúan objetivos observados y predicciones disponibles.
    for nombre, columna_pred in [
        ("Persistencia", "pred_persistencia"),
        ("Persistencia diaria", "pred_diaria"),
    ]:
        evaluables = parte[["o3", columna_pred]].dropna()

        # Evita calcular métricas si no existen suficientes observaciones.
        if len(evaluables) < 2:
            print(f"Advertencia: pocos datos para {nombre} en {particion}.")
            continue

        real = evaluables["o3"].to_numpy()
        predicho = evaluables[columna_pred].to_numpy()

        # Calcula métricas de error en las mismas unidades que el ozono: µg/m³.
        mae = mean_absolute_error(real, predicho)
        rmse = np.sqrt(mean_squared_error(real, predicho))
        r2 = r2_score(real, predicho)

        resultados.append({
            "particion": particion,
            "metodo": nombre,
            "n_evaluaciones": len(evaluables),
            "MAE_ug_m3": mae,
            "RMSE_ug_m3": rmse,
            "R2": r2,
        })

# Presenta y guarda los resultados de referencia.
tabla = pd.DataFrame(resultados)

if tabla.empty:
    raise ValueError("No se pudieron calcular métricas.")

print("\n=== LÍNEAS BASE DE PREDICCIÓN DE O3 ===")
print(tabla.round(4).to_string(index=False))

salida = Path("outputs/tables/o3_baseline_metrics.csv")
salida.parent.mkdir(parents=True, exist_ok=True)
tabla.to_csv(salida, index=False)

print(f"\nResultados guardados en: {salida}")
