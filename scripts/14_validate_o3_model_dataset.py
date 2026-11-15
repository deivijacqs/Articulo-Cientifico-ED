
from pathlib import Path
import pandas as pd

# Define la ruta del conjunto horario generado por el script 13.
RUTA = Path("outputs/tables/o3_dataset_horario.csv")

# Carga el CSV y convierte la columna temporal correcta a fechas UTC.
df = pd.read_csv(RUTA, parse_dates=["fecha_utc"])

# Comprueba que existan las columnas necesarias para la auditoría.
columnas_requeridas = [
    "fecha_utc",
    "valor_o3_original",
    "valor_o3_modelo",
    "observacion_api",
    "valor_numerico_disponible",
    "bandera_activa",
    "observacion_entrenable",
]

faltantes = [col for col in columnas_requeridas if col not in df.columns]

# Detiene la ejecución si falta alguna columna importante.
if faltantes:
    raise ValueError(f"Faltan columnas necesarias: {faltantes}")

# Ordena las observaciones por fecha.
df = df.sort_values("fecha_utc").reset_index(drop=True)

print("\n=== VALIDACIÓN DEL CONJUNTO HORARIO DE O3 ===")

# Informa el periodo y el tamaño del conjunto de datos.
print(f"Filas: {len(df)}")
print(f"Fecha inicial: {df['fecha_utc'].min()}")
print(f"Fecha final: {df['fecha_utc'].max()}")

# Comprueba fechas duplicadas o ausentes.
print(f"Fechas duplicadas: {df['fecha_utc'].duplicated().sum()}")
print(f"Fechas nulas: {df['fecha_utc'].isna().sum()}")

# Comprueba que los registros consecutivos estén separados por una hora.
diferencias = df["fecha_utc"].diff().dropna()
intervalos_no_horarios = (
    diferencias != pd.Timedelta(hours=1)
).sum()

print(f"Intervalos distintos de una hora: {intervalos_no_horarios}")

# Convierte las columnas de control a valores booleanos.
columnas_booleanas = [
    "observacion_api",
    "valor_numerico_disponible",
    "bandera_activa",
    "observacion_entrenable",
]

for columna in columnas_booleanas:
    df[columna] = df[columna].fillna(False).astype(bool)

# Resume los registros disponibles y los que pueden utilizarse provisionalmente.
for columna in columnas_booleanas:
    print(f"{columna}: {int(df[columna].sum())}")

# Comprueba que la máscara de entrenamiento sea coherente con los valores.
valores_modelo = pd.to_numeric(
    df["valor_o3_modelo"], errors="coerce"
)

entrenables = df["observacion_entrenable"]

print(
    "Observaciones entrenables sin valor numérico:",
    int((entrenables & valores_modelo.isna()).sum()),
)

print(
    "Valores del modelo fuera de la máscara de entrenamiento:",
    int((~entrenables & valores_modelo.notna()).sum()),
)

# Busca concentraciones negativas en los datos originales y en los de entrenamiento.
valores_originales = pd.to_numeric(
    df["valor_o3_original"], errors="coerce"
)

print(
    "Valores originales negativos:",
    int((valores_originales < 0).sum()),
)

print(
    "Valores negativos en el conjunto de entrenamiento:",
    int((valores_modelo < 0).sum()),
)

# Calcula la cobertura por año y mes usando las horas de la cuadrícula.
df["anio"] = df["fecha_utc"].dt.year
df["mes"] = df["fecha_utc"].dt.month

resumen_mensual = df.groupby(["anio", "mes"]).agg(
    horas_en_cuadricula=("fecha_utc", "size"),
    horas_con_registro=("observacion_api", "sum"),
    horas_entrenables=("observacion_entrenable", "sum"),
)

# Expresa la cantidad de horas entrenables como porcentaje de cada mes.
resumen_mensual["porcentaje_entrenable"] = (
    100
    * resumen_mensual["horas_entrenables"]
    / resumen_mensual["horas_en_cuadricula"]
)

print("\n--- 12 meses con menor porcentaje entrenable ---")
print(
    resumen_mensual.sort_values("porcentaje_entrenable")
    .head(12)
    .round(2)
    .to_string()
)

# Guarda la auditoría mensual para documentar la calidad temporal de los datos.
salida = Path("outputs/tables/o3_validacion_mensual.csv")
salida.parent.mkdir(parents=True, exist_ok=True)
resumen_mensual.to_csv(salida)

print(f"\nAuditoría mensual guardada en: {salida}")
print("\nValidación terminada.")
