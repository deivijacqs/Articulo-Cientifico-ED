from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = Path("outputs/tables/31_dependencia_temporal_agregada.csv")
CARPETA_SALIDA = Path("outputs/figures")

# Se representan los primeros 12 retardos.
MAX_LAG = 12

HORIZONTES = [1, 6, 24]

METRICAS = {
    "MAE": "Difference in absolute error",
    "MSE": "Difference in squared error"
}


# ============================================================
# CARGA Y VALIDACIÓN
# ============================================================

df = pd.read_csv(ARCHIVO)

columnas_requeridas = {
    "horizonte",
    "metrica",
    "lag",
    "autocorrelacion",
    "pares",
    "segmentos"
}

if not columnas_requeridas.issubset(df.columns):
    raise ValueError("El archivo no contiene todas las columnas necesarias.")

df = df[
    (df["lag"] <= MAX_LAG) &
    (df["horizonte"].isin(HORIZONTES))
].copy()


# ============================================================
# GENERACIÓN DE LA FIGURA
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4.8),
    sharey=True
)

for ax, horizonte in zip(axes, HORIZONTES):

    datos_h = df[df["horizonte"] == horizonte]

    for metrica in METRICAS:

        datos = datos_h[
            datos_h["metrica"] == metrica
        ].sort_values("lag")

        ax.plot(
            datos["lag"],
            datos["autocorrelacion"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=metrica
        )

    # Referencia de autocorrelación nula.
    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="black"
    )

    ax.set_title(f"{horizonte}-hour horizon")
    ax.set_xlabel("Lag (hours)")
    ax.set_xticks(range(1, MAX_LAG + 1))
    ax.grid(alpha=0.25)

axes[0].set_ylabel("Autocorrelation of paired loss differences")

axes[-1].legend(
    title="Loss metric",
    frameon=False
)

fig.suptitle(
    "Temporal autocorrelation of paired model-loss differences",
    fontsize=13,
    y=1.03
)

fig.tight_layout()

# ============================================================
# EXPORTACIÓN
# ============================================================

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

ruta_png = CARPETA_SALIDA / "fig_04_autocorrelacion_diferencias_perdida.png"
ruta_pdf = CARPETA_SALIDA / "fig_04_autocorrelacion_diferencias_perdida.pdf"

fig.savefig(
    ruta_png,
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    ruta_pdf,
    bbox_inches="tight"
)

plt.close(fig)

print("FIGURA GENERADA CORRECTAMENTE")
print(f"PNG: {ruta_png}")
print(f"PDF: {ruta_pdf}")
print(f"Retardos representados: 1-{MAX_LAG} horas")