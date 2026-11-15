
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. RUTAS
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]

ARCHIVO_ENTRADA = (
    RAIZ / "outputs" / "tables" /
    "o3_pinn_comparacion_metricas.csv"
)

CARPETA_SALIDA = RAIZ / "outputs" / "figures"
CARPETA_TABLAS = RAIZ / "outputs" / "tables"

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGAR Y FILTRAR LOS RESULTADOS
# ============================================================

df = pd.read_csv(ARCHIVO_ENTRADA)

fisico = df[
    df["modelo"] == "Modelo_neuronal_fisico"
].copy()

# Comprobar que existen los tres horizontes.
horizontes_esperados = {1, 6, 24}

if set(fisico["horizonte_horas"].unique()) != horizontes_esperados:
    raise ValueError(
        "Los horizontes disponibles no coinciden con 1, 6 y 24 horas."
    )

# Comprobar que k es consistente entre validación y test.
consistencia = (
    fisico.groupby("horizonte_horas")["k_por_hora"]
    .nunique()
)

if (consistencia != 1).any():
    raise ValueError(
        "Se encontraron valores diferentes de k entre particiones."
    )

# Extraer un registro por horizonte.
parametros = (
    fisico.groupby("horizonte_horas", as_index=False)
    .agg(
        k_por_hora=("k_por_hora", "first"),
        peso_fisico=("peso_fisico", "first")
    )
    .sort_values("horizonte_horas")
)

# Calcular la escala temporal característica.
parametros["escala_temporal_horas"] = (
    1.0 / parametros["k_por_hora"]
)

parametros["horizonte"] = (
    parametros["horizonte_horas"].astype(str) + " h"
)


# ============================================================
# 3. GUARDAR LA TABLA DE PARÁMETROS
# ============================================================

archivo_tabla = (
    CARPETA_TABLAS / "45_parametros_dinamicos_fisicos.csv"
)

parametros.to_csv(
    archivo_tabla,
    index=False,
    float_format="%.8f"
)


# ============================================================
# 4. CONSTRUIR LA FIGURA
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10, 4.8)
)

x = np.arange(len(parametros))
etiquetas = parametros["horizonte"].tolist()

valores_k = parametros["k_por_hora"].to_numpy()
valores_tau = parametros["escala_temporal_horas"].to_numpy()


# ------------------------------------------------------------
# Panel A: coeficiente de atenuación k
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    x,
    valores_k,
    marker="o",
    markersize=8,
    linewidth=1.8
)

for i, valor in enumerate(valores_k):
    ax.annotate(
        f"{valor:.5f}",
        (x[i], valor),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

ax.set_xticks(x)
ax.set_xticklabels(etiquetas)

ax.set_xlabel("Forecasting horizon")
ax.set_ylabel(r"Learned attenuation coefficient, $k$ (h$^{-1}$)")

ax.set_title("(a) Learned attenuation coefficient")

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)


# ------------------------------------------------------------
# Panel B: escala temporal característica 1/k
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    x,
    valores_tau,
    marker="o",
    markersize=8,
    linewidth=1.8
)

for i, valor in enumerate(valores_tau):
    ax.annotate(
        f"{valor:.2f}",
        (x[i], valor),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

ax.set_xticks(x)
ax.set_xticklabels(etiquetas)

ax.set_xlabel("Forecasting horizon")
ax.set_ylabel(r"Characteristic timescale, $1/k$ (h)")

ax.set_title("(b) Derived characteristic timescale")

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)


# ============================================================
# 5. FORMATO Y EXPORTACIÓN
# ============================================================

fig.tight_layout()

archivo_png = (
    CARPETA_SALIDA /
    "fig_05_parametros_dinamicos_fisicos.png"
)

archivo_pdf = (
    CARPETA_SALIDA /
    "fig_05_parametros_dinamicos_fisicos.pdf"
)

fig.savefig(
    archivo_png,
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    archivo_pdf,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 6. INFORME DE RESULTADOS
# ============================================================

print("\nFIGURA GENERADA CORRECTAMENTE")
print("=" * 65)

print("\nPARÁMETROS FÍSICOS APRENDIDOS:")
print(
    parametros[
        [
            "horizonte_horas",
            "k_por_hora",
            "escala_temporal_horas",
            "peso_fisico"
        ]
    ].to_string(index=False)
)

print("\nARCHIVOS GENERADOS:")
print("PNG:", archivo_png.relative_to(RAIZ))
print("PDF:", archivo_pdf.relative_to(RAIZ))
print("CSV:", archivo_tabla.relative_to(RAIZ))
