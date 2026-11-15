
import pandas as pd

# Cargar las predicciones detalladas generadas por el script.
ruta = "outputs/tables/o3_pinn_predicciones_detalladas.csv"
df = pd.read_csv(ruta, parse_dates=["fecha_objetivo"])

# Revisar las dimensiones y las primeras filas.
print("Dimensiones:", df.shape)
print("\nPrimeras filas:")
print(df.head())

# Comprobar las columnas disponibles.
print("\nColumnas:")
print(df.columns.tolist())

# Contar predicciones por horizonte, partición y modelo.
print("\nCantidad de predicciones:")
print(
    df.groupby(
        ["horizonte_horas", "particion", "modelo"]
    ).size()
)

# Comprobar valores ausentes.
print("\nValores ausentes por columna:")
print(df.isnull().sum())

# Comprobar si existen duplicados de predicciones.
claves = [
    "fecha_objetivo",
    "horizonte_horas",
    "particion",
    "modelo",
]
print("\nFilas duplicadas:", df.duplicated(claves).sum())
