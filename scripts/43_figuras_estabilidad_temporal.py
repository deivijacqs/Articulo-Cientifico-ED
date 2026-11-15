
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

ARCHIVO_RESUMEN = Path(
    "outputs/tables/35_resumen_estabilidad_subsampling.csv"
)

ARCHIVO_SEGMENTOS = Path(
    "outputs/tables/35_diagnostico_segmentos_subsampling.csv"
)

CARPETA_SALIDA = Path("outputs/figures")

# Archivos de salida
FIGURA_02_PNG = (
    CARPETA_SALIDA / "fig_02_estabilidad_ventanas_temporales.png"
)

FIGURA_02_PDF = (
    CARPETA_SALIDA / "fig_02_estabilidad_ventanas_temporales.pdf"
)

FIGURA_03_PNG = (
    CARPETA_SALIDA / "fig_03_influencia_segmentos_temporales.png"
)

FIGURA_03_PDF = (
    CARPETA_SALIDA / "fig_03_influencia_segmentos_temporales.pdf"
)


# ============================================================
# 2. CARGAR Y VALIDAR LOS RESULTADOS
# ============================================================

resumen = pd.read_csv(ARCHIVO_RESUMEN)
segmentos = pd.read_csv(ARCHIVO_SEGMENTOS)

CARPETA_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

columnas_resumen = [
    "horizonte",
    "bloque_horas",
    "cobertura_pct",
    "delta_mae_elegibles",
    "mediana_ventanas_mae",
    "ventanas_mae_p05",
    "ventanas_mae_p95"
]

faltantes = [
    columna
    for columna in columnas_resumen
    if columna not in resumen.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas en el resumen: {faltantes}"
    )

columnas_segmentos = [
    "horizonte",
    "bloque_horas",
    "id_segmento",
    "fecha_inicio",
    "elegible",
    "cambio_delta_mae"
]

faltantes = [
    columna
    for columna in columnas_segmentos
    if columna not in segmentos.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas en el diagnóstico: {faltantes}"
    )

# Convertir fechas a formato temporal.
segmentos["fecha_inicio"] = pd.to_datetime(
    segmentos["fecha_inicio"],
    utc=True
)

# Asegurar que la columna de elegibilidad sea booleana.
if segmentos["elegible"].dtype != bool:
    segmentos["elegible"] = (
        segmentos["elegible"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

# Comprobar que los resultados necesarios no estén vacíos.
if resumen.empty or segmentos.empty:
    raise ValueError(
        "Los archivos de resultados están vacíos."
    )


# ============================================================
# 3. FIGURA 2: ESTABILIDAD ENTRE VENTANAS TEMPORALES
# ============================================================

horizontes = [1, 6, 24]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5.5),
    sharey=False
)

for ax, horizonte in zip(axes, horizontes):

    datos = resumen[
        resumen["horizonte"] == horizonte
    ].sort_values("bloque_horas")

    if datos.empty:
        raise ValueError(
            f"No hay resultados para el horizonte {horizonte} h."
        )

    x = np.arange(len(datos))

    mediana = datos["mediana_ventanas_mae"].to_numpy()
    inferior = datos["ventanas_mae_p05"].to_numpy()
    superior = datos["ventanas_mae_p95"].to_numpy()
    observado = datos["delta_mae_elegibles"].to_numpy()

    # Comprobar que los percentiles estén correctamente ordenados.
    if np.any(inferior > superior):
        raise ValueError(
            f"Percentiles inconsistentes en horizonte {horizonte} h."
        )

    # Rango descriptivo P5-P95 de las diferencias entre ventanas.
    ax.errorbar(
        x,
        mediana,
        yerr=[
            mediana - inferior,
            superior - mediana
        ],
        fmt="o",
        markersize=7,
        capsize=5,
        linewidth=1.7,
        label="Window median and P5-P95 range"
    )

    # Diferencia global observada sobre los segmentos elegibles.
    ax.scatter(
        x,
        observado,
        marker="D",
        s=65,
        zorder=4,
        label="Observed difference"
    )

    # Referencia de ausencia de diferencia.
    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.2
    )

    ax.set_title(
        f"{horizonte}-hour forecast",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("Temporal window length (hours)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        datos["bloque_horas"].astype(str)
    )

    ax.grid(
        True,
        linestyle=":",
        alpha=0.5
    )

    # Mostrar la cobertura de las observaciones en cada bloque.
    for posicion, (_, fila) in enumerate(datos.iterrows()):

        ax.annotate(
            f"{fila['cobertura_pct']:.1f}%",
            (posicion, 0.97),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=8
        )

axes[0].set_ylabel(
    "Difference in MAE (µg/m³)"
)

# Leyenda compartida.
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, -0.015)
)

fig.suptitle(
    "Temporal stability of paired MAE differences",
    fontsize=15,
    fontweight="bold",
    y=1.02
)

fig.tight_layout(
    rect=[0, 0.10, 1, 0.95]
)

fig.savefig(
    FIGURA_02_PNG,
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    FIGURA_02_PDF,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 4. FIGURA 3: INFLUENCIA DE SEGMENTOS TEMPORALES
# ============================================================

# Para cada horizonte se utiliza la mayor longitud de bloque
# evaluada en el análisis de estabilidad.
bloques_representativos = {
    1: 6,
    6: 6,
    24: 12
}

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5.5),
    sharey=False
)

for ax, horizonte in zip(axes, horizontes):

    bloque = bloques_representativos[horizonte]

    datos = segmentos[
        (segmentos["horizonte"] == horizonte)
        & (segmentos["bloque_horas"] == bloque)
        & (segmentos["elegible"] == True)
    ].copy()

    datos = datos.dropna(
        subset=[
            "fecha_inicio",
            "cambio_delta_mae"
        ]
    ).sort_values("fecha_inicio")

    if datos.empty:
        raise ValueError(
            f"No hay segmentos válidos para {horizonte} h."
        )

    # Cambio de la diferencia global al retirar cada segmento.
    ax.scatter(
        datos["fecha_inicio"],
        datos["cambio_delta_mae"],
        s=28,
        alpha=0.75
    )

    # Línea de ausencia de influencia.
    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.2
    )

    # Identificar el segmento con mayor influencia absoluta.
    indice_mayor = (
        datos["cambio_delta_mae"]
        .abs()
        .idxmax()
    )

    mayor = datos.loc[indice_mayor]

    ax.annotate(
        f"Segment {int(mayor['id_segmento'])}",
        (
            mayor["fecha_inicio"],
            mayor["cambio_delta_mae"]
        ),
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=8
    )

    ax.set_title(
        f"{horizonte}-hour forecast",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("Segment start date")

    ax.grid(
        True,
        linestyle=":",
        alpha=0.5
    )

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=2)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b\n%Y")
    )

axes[0].set_ylabel(
    "Change in global ΔMAE after segment removal (µg/m³)"
)

fig.suptitle(
    "Influence of individual temporal segments on paired MAE differences",
    fontsize=14,
    fontweight="bold",
    y=1.02
)

fig.tight_layout()

fig.savefig(
    FIGURA_03_PNG,
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    FIGURA_03_PDF,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 5. CONFIRMACIÓN
# ============================================================

print("\nFIGURAS GENERADAS CORRECTAMENTE")
print("=" * 75)

print("\nFIGURA 2")
print("PNG:", FIGURA_02_PNG)
print("PDF:", FIGURA_02_PDF)

print("\nFIGURA 3")
print("PNG:", FIGURA_03_PNG)
print("PDF:", FIGURA_03_PDF)

print("\nResolución: 400 dpi")
print("Figura 2: tres horizontes y tres longitudes de ventana.")
print("Figura 3: segmentos elegibles para el bloque más largo de cada horizonte.")
