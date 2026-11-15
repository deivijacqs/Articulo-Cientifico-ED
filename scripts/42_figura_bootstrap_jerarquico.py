
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = Path(
    "outputs/tables/33_bootstrap_jerarquico_bloques.csv"
)

CARPETA_SALIDA = Path("outputs/figures")

ARCHIVO_PNG = CARPETA_SALIDA / (
    "fig_01_bootstrap_jerarquico_mae.png"
)

ARCHIVO_PDF = CARPETA_SALIDA / (
    "fig_01_bootstrap_jerarquico_mae.pdf"
)

# ============================================================
# CARGA Y VALIDACIÓN
# ============================================================

df = pd.read_csv(ARCHIVO)

# Conservar exclusivamente la diferencia de MAE.
df = df[df["metrica"] == "delta_mae"].copy()

columnas = [
    "horizonte",
    "bloque_horas",
    "observaciones",
    "replicas",
    "observado",
    "media_bootstrap",
    "ic95_inferior",
    "ic95_superior"
]

df = df[columnas].sort_values(
    ["horizonte", "bloque_horas"]
)

if df.empty:
    raise ValueError("No hay resultados de delta_mae.")

if df[["media_bootstrap", "ic95_inferior",
       "ic95_superior", "observado"]].isna().any().any():
    raise ValueError("Existen valores ausentes en los resultados.")

if (
    (df["ic95_inferior"] > df["media_bootstrap"])
    | (df["ic95_superior"] < df["media_bootstrap"])
).any():
    raise ValueError(
        "La media bootstrap está fuera de su intervalo."
    )

# ============================================================
# CONFIGURACIÓN DE LA FIGURA
# ============================================================

CARPETA_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

horizontes = [1, 6, 24]

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(15, 5.5)
)

# ============================================================
# CONSTRUIR LOS TRES PANELES
# ============================================================

for ax, horizonte in zip(axes, horizontes):

    datos = df[df["horizonte"] == horizonte].copy()

    if datos.empty:
        raise ValueError(
            f"No existen resultados para {horizonte} horas."
        )

    x = datos["bloque_horas"].to_numpy()

    media = datos["media_bootstrap"].to_numpy()
    inferior = datos["ic95_inferior"].to_numpy()
    superior = datos["ic95_superior"].to_numpy()

    observado = datos["observado"].to_numpy()

    # Intervalos asimétricos alrededor de la media bootstrap.
    errores = [
        media - inferior,
        superior - media
    ]

    # Media bootstrap e intervalo percentil del 95 %.
    ax.errorbar(
        x,
        media,
        yerr=errores,
        fmt="o",
        capsize=5,
        markersize=7,
        linewidth=1.5,
        label="Bootstrap mean and 95% interval"
    )

    # Diferencia observada en los datos originales.
    ax.scatter(
        x,
        observado,
        marker="D",
        s=65,
        zorder=3,
        label="Observed difference"
    )

    # Referencia de igualdad entre modelos.
    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        f"{horizonte}-hour forecast",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Temporal block length (hours)",
        fontsize=10
    )

    ax.set_ylabel(
        "Difference in MAE (µg/m³)",
        fontsize=10
    )

    ax.set_xticks(x)

    ax.grid(
        True,
        linestyle=":",
        alpha=0.5
    )

    ax.tick_params(
        axis="both",
        labelsize=9
    )

# ============================================================
# LEYENDA GENERAL
# ============================================================

elementos_leyenda = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="-",
        label="Bootstrap mean and 95% percentile interval"
    ),
    Line2D(
        [0], [0],
        marker="D",
        linestyle="None",
        label="Observed difference"
    ),
    Line2D(
        [0], [0],
        linestyle="--",
        label="Zero difference"
    )
]

fig.legend(
    handles=elementos_leyenda,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=9
)

fig.suptitle(
    "Robustness of paired MAE differences under hierarchical block bootstrap",
    fontsize=13,
    fontweight="bold",
    y=1.02
)

fig.tight_layout(
    rect=[0, 0.12, 1, 0.96]
)

# ============================================================
# EXPORTACIÓN
# ============================================================

fig.savefig(
    ARCHIVO_PNG,
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    ARCHIVO_PDF,
    bbox_inches="tight"
)

plt.close(fig)

print("\nFIGURA GENERADA CORRECTAMENTE")
print("=" * 65)
print("PNG:", ARCHIVO_PNG)
print("PDF:", ARCHIVO_PDF)
print("Resolución: 400 dpi")
print("\nConfiguraciones representadas:", len(df))
