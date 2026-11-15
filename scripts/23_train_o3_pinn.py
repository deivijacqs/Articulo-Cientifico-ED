
# scripts/23_train_o3_pinn.py
# ============================================================
# ComparaciÃ³n de modelos para pronosticar la concentraciÃ³n de O3:
# 1. Persistencia.
# 2. MLP sin meteorologÃ­a.
# 3. MLP con meteorologÃ­a.
# 4. Modelo neuronal con estructura fÃ­sica.
#
# EcuaciÃ³n dinÃ¡mica simplificada:
#       dC/dt = q_theta(x_t) - k*C(t),  k > 0
#
# q_theta es una fuente efectiva aprendida, no una estimaciÃ³n
# directa de producciÃ³n quÃ­mica. La ecuaciÃ³n no representa toda
# la quÃ­mica ni el transporte atmosfÃ©rico del ozono.
#
# El modelo fÃ­sico supone que q_theta permanece constante
# durante el horizonte de predicciÃ³n.
#
# Buenas prÃ¡cticas:
# - DivisiÃ³n cronolÃ³gica preexistente, sin reorganizar particiones.
# - Escaladores ajustados solo con train.
# - SelecciÃ³n de Ã©poca con validation.
# - Test no participa en el entrenamiento ni en la selecciÃ³n.
# - MeteorologÃ­a del origen t; calendario conocido del objetivo t+h.
# - No se rellenan artificialmente las concentraciones faltantes.
# ============================================================

from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ------------------------------------------------------------
# 1. ConfiguraciÃ³n
# ------------------------------------------------------------

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_DATOS = (
    RUTA_PROYECTO
    / "outputs"
    / "tables"
    / "o3_dataset_meteorologia.csv"
)
RUTA_SALIDA = RUTA_PROYECTO / "outputs" / "tables"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
HORIZONTES = [1, 6, 24]

MAX_EPOCAS = 1500
PACIENCIA = 60
TASA_APRENDIZAJE = 0.001

# Peso del residuo de la ecuaciÃ³n diferencial.
# No conviene ajustarlo usando las mÃ©tricas del test.
PESO_FISICO = 0.1

torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)
torch.set_num_threads(1)

DISPOSITIVO = torch.device("cpu")


# ------------------------------------------------------------
# 2. Cargar y validar los datos
# ------------------------------------------------------------

if not RUTA_DATOS.exists():
    raise FileNotFoundError(
        f"No se encontrÃ³ el archivo de entrada:\n{RUTA_DATOS}"
    )

datos = pd.read_csv(
    RUTA_DATOS,
    parse_dates=["fecha_utc"],
)

datos = (
    datos.sort_values("fecha_utc")
    .reset_index(drop=True)
)

columnas_requeridas = [
    "fecha_utc",
    "valor_o3_modelo",
    "particion",
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos",
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

faltantes = [
    columna
    for columna in columnas_requeridas
    if columna not in datos.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas requeridas en el CSV: {faltantes}"
    )

# Asegurar que las fechas sean vÃ¡lidas y que no existan duplicados.
if datos["fecha_utc"].isna().any():
    raise ValueError("Hay fechas UTC vacÃ­as en el archivo.")

if datos["fecha_utc"].duplicated().any():
    raise ValueError(
        "Hay fechas duplicadas. Revisa la consolidaciÃ³n de la serie."
    )

# La serie debe estar en una cuadrÃ­cula horaria continua.
diferencias_horas = (
    datos["fecha_utc"]
    .diff()
    .dropna()
    .dt.total_seconds()
    / 3600
)

if not np.allclose(diferencias_horas.to_numpy(), 1.0):
    raise ValueError(
        "La serie no es horaria y continua. "
        "No se deben aplicar desplazamientos horarios "
        "sin revisar primero la cuadrÃ­cula temporal."
    )

# Usar la mÃ¡scara de calidad previamente auditada.
# No se modifica el criterio de calidad en este script.
if "observacion_entrenable" in datos.columns:
    columna_entrenable = datos["observacion_entrenable"]

    if columna_entrenable.dtype == bool:
        mascara_entrenable = columna_entrenable.copy()
    else:
        mascara_entrenable = (
            columna_entrenable.astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "si", "sÃ­", "yes"])
        )
else:
    raise ValueError(
        "No existe 'observacion_entrenable'. "
        "Para mantener el criterio de calidad de la auditorÃ­a, "
        "no se aplicarÃ¡ automÃ¡ticamente una mÃ¡scara alternativa."
    )

datos["C"] = pd.to_numeric(
    datos["valor_o3_modelo"],
    errors="coerce",
)

datos["valido"] = (
    mascara_entrenable
    & datos["C"].notna()
    & np.isfinite(datos["C"])
)

# Convertir las variables de entrada a valores numÃ©ricos.
variables_meteorologicas = [
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

variables_tiempo_origen = [
    "hora_sin",
    "hora_cos",
    "dia_anio_sin",
    "dia_anio_cos",
]

for columna in variables_meteorologicas + variables_tiempo_origen:
    datos[columna] = pd.to_numeric(
        datos[columna],
        errors="coerce",
    )


# ------------------------------------------------------------
# 3. Arquitectura de los modelos
# ------------------------------------------------------------

class MLP(nn.Module):
    """MLP convencional para predecir la concentraciÃ³n futura."""

    def __init__(self, numero_entradas):
        super().__init__()

        self.red = nn.Sequential(
            nn.Linear(numero_entradas, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.red(x).squeeze(-1)


class ModeloFisico(nn.Module):
    """
    Modelo con soluciÃ³n analÃ­tica de:
        dC/dt = q_theta(x) - k*C

    q_theta(x) es una fuente efectiva no negativa.
    k es estrictamente positivo.

    La soluciÃ³n supone q constante durante el horizonte h:
        C(t+h) = exp(-k*h)*C(t)
                 + q/k * (1 - exp(-k*h))
    """

    def __init__(self, numero_entradas):
        super().__init__()

        self.fuente = nn.Sequential(
            nn.Linear(numero_entradas, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # InicializaciÃ³n de k en un valor positivo pequeÃ±o.
        self.raw_k = nn.Parameter(torch.tensor(-3.0))

    def obtener_k(self):
        # softplus garantiza k > 0.
        return (
            torch.nn.functional.softplus(self.raw_k)
            + 1e-4
        )

    def obtener_fuente(self, x):
        # Fuente efectiva no negativa.
        return torch.nn.functional.softplus(
            self.fuente(x).squeeze(-1)
        )

    def predecir(self, x, c_actual, horizonte):
        k = self.obtener_k()
        q = self.obtener_fuente(x)

        # SoluciÃ³n analÃ­tica de la ecuaciÃ³n diferencial.
        factor_decaimiento = torch.exp(-k * horizonte)

        # CÃ¡lculo numÃ©ricamente estable de 1 - exp(-k*h).
        factor_fuente = -torch.expm1(-k * horizonte) / k

        return (
            factor_decaimiento * c_actual
            + factor_fuente * q
        )

    def residuo_una_hora(
        self,
        x,
        c_actual,
        c_siguiente,
    ):
        # Residuo de Euler para el intervalo de una hora:
        # C(t+1)-C(t) - [q(x)-k*C(t)]*1 hora.
        k = self.obtener_k()
        q = self.obtener_fuente(x)

        derivada = q - k * c_actual

        return (
            c_siguiente
            - c_actual
            - derivada
        )


# ------------------------------------------------------------
# 4. Funciones de entrenamiento
# ------------------------------------------------------------

def convertir_tensor(valores):
    """Convierte datos a tensores float32 para PyTorch."""

    return torch.as_tensor(
        np.asarray(valores, dtype=np.float32),
        dtype=torch.float32,
        device=DISPOSITIVO,
    )


def entrenar_mlp(
    x_train,
    y_train,
    x_val,
    y_val,
):
    """
    Entrena un MLP.
    La mejor Ã©poca se selecciona usando el error cuadrÃ¡tico
    medio de validaciÃ³n, nunca con el test.
    """

    modelo = MLP(x_train.shape[1]).to(DISPOSITIVO)

    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=TASA_APRENDIZAJE,
    )

    funcion_perdida = nn.MSELoss()

    mejor_estado = copy.deepcopy(modelo.state_dict())
    mejor_perdida = float("inf")
    mejor_epoca = 0
    epocas_sin_mejora = 0

    for epoca in range(1, MAX_EPOCAS + 1):
        modelo.train()
        optimizador.zero_grad()

        pred_train = modelo(x_train)
        perdida_train = funcion_perdida(
            pred_train,
            y_train,
        )

        perdida_train.backward()
        optimizador.step()

        modelo.eval()

        with torch.no_grad():
            perdida_val = funcion_perdida(
                modelo(x_val),
                y_val,
            ).item()

        if perdida_val < mejor_perdida:
            mejor_perdida = perdida_val
            mejor_estado = copy.deepcopy(
                modelo.state_dict()
            )
            mejor_epoca = epoca
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1

        if epocas_sin_mejora >= PACIENCIA:
            break

    modelo.load_state_dict(mejor_estado)

    return modelo, mejor_epoca, mejor_perdida


def entrenar_modelo_fisico(
    x_train,
    c_train,
    y_train,
    x_val,
    c_val,
    y_val,
    x_residuo_train,
    c_residuo_train,
    c_siguiente_train,
    x_residuo_val,
    c_residuo_val,
    c_siguiente_val,
    horizonte,
):
    """
    Combina:
    1. Error de pronÃ³stico a horizonte h.
    2. Error del residuo diferencial de transiciones de una hora.

    El residuo diferencial solo se calcula para transiciones vÃ¡lidas.
    """

    modelo = ModeloFisico(
        x_train.shape[1]
    ).to(DISPOSITIVO)

    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=TASA_APRENDIZAJE,
    )

    mejor_estado = copy.deepcopy(
        modelo.state_dict()
    )
    mejor_perdida = float("inf")
    mejor_epoca = 0
    epocas_sin_mejora = 0

    for epoca in range(1, MAX_EPOCAS + 1):
        modelo.train()
        optimizador.zero_grad()

        # Error de pronÃ³stico en el horizonte seleccionado.
        pred_train = modelo.predecir(
            x_train,
            c_train,
            horizonte,
        )

        perdida_pronostico = torch.mean(
            (pred_train - y_train) ** 2
        )

        # Residuo fÃ­sico en transiciones vÃ¡lidas de una hora.
        residuo_train = modelo.residuo_una_hora(
            x_residuo_train,
            c_residuo_train,
            c_siguiente_train,
        )

        perdida_fisica = torch.mean(
            residuo_train ** 2
        )

        # Objetivo combinado de entrenamiento.
        perdida_total = (
            perdida_pronostico
            + PESO_FISICO * perdida_fisica
        )

        perdida_total.backward()
        optimizador.step()

        # ValidaciÃ³n para seleccionar la Ã©poca.
        modelo.eval()

        with torch.no_grad():
            pred_val = modelo.predecir(
                x_val,
                c_val,
                horizonte,
            )

            perdida_val_pronostico = torch.mean(
                (pred_val - y_val) ** 2
            )

            residuo_val = modelo.residuo_una_hora(
                x_residuo_val,
                c_residuo_val,
                c_siguiente_val,
            )

            perdida_val_fisica = torch.mean(
                residuo_val ** 2
            )

            perdida_val_total = (
                perdida_val_pronostico
                + PESO_FISICO * perdida_val_fisica
            ).item()

        if perdida_val_total < mejor_perdida:
            mejor_perdida = perdida_val_total
            mejor_estado = copy.deepcopy(
                modelo.state_dict()
            )
            mejor_epoca = epoca
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1

        if epocas_sin_mejora >= PACIENCIA:
            break

    modelo.load_state_dict(mejor_estado)

    return modelo, mejor_epoca, mejor_perdida


def calcular_metricas(y_real, y_pred):
    """Devuelve mÃ©tricas de regresiÃ³n en las unidades originales."""

    return {
        "n": int(len(y_real)),
        "MAE": float(
            mean_absolute_error(y_real, y_pred)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_real, y_pred)
            )
        ),
        "R2": float(
            r2_score(y_real, y_pred)
        ),
    }


# ------------------------------------------------------------
# 5. DefiniciÃ³n de las caracterÃ­sticas
# ------------------------------------------------------------

# Los MLP usan el calendario del instante objetivo t+h.
variables_tiempo_objetivo = [
    "hora_sin_objetivo",
    "hora_cos_objetivo",
    "dia_anio_sin_objetivo",
    "dia_anio_cos_objetivo",
]

# El modelo fÃ­sico usa calendario y meteorologÃ­a del origen t.
# La concentraciÃ³n actual se incorpora directamente en la ecuaciÃ³n.
features_fuente = (
    ["C_lag24"]
    + variables_tiempo_origen
    + variables_meteorologicas
)


# ------------------------------------------------------------
# 6. Entrenar y evaluar cada horizonte
# ------------------------------------------------------------

resultados = []
resumen_entrenamiento = []
# Almacena predicciones por fecha objetivo, horizonte, partición y modelo.
predicciones_detalladas = []

for horizonte in HORIZONTES:
    print("\n" + "=" * 68)
    print(
        f"ENTRENAMIENTO PARA HORIZONTE DE {horizonte} HORA(S)"
    )
    print("=" * 68)

    muestra = datos.copy()

    # ConcentraciÃ³n disponible al emitir el pronÃ³stico.
    muestra["C_actual"] = muestra["C"]

    # ConcentraciÃ³n disponible 24 horas antes del origen.
    muestra["C_lag24"] = muestra["C"].shift(24)

    # Objetivo de pronÃ³stico a t+h.
    muestra["C_objetivo"] = muestra["C"].shift(-horizonte)

    # ConcentraciÃ³n de la hora inmediatamente posterior al origen.
    muestra["C_siguiente"] = muestra["C"].shift(-1)

    # Particiones de los tiempos de destino y de la transiciÃ³n.
    muestra["particion_objetivo"] = (
        muestra["particion"].shift(-horizonte)
    )

    muestra["particion_siguiente"] = (
        muestra["particion"].shift(-1)
    )

    # Calendario del instante objetivo.
    # El desplazamiento negativo toma el calendario de t+h
    # y lo asigna a la fila que representa el origen t.
    for origen, objetivo in zip(
        variables_tiempo_origen,
        variables_tiempo_objetivo,
    ):
        muestra[objetivo] = (
            datos[origen].shift(-horizonte)
        )

    # Entradas de los dos MLP.
    features_mlp_sin = (
        ["C_actual", "C_lag24"]
        + variables_tiempo_objetivo
    )

    features_mlp_con = (
        features_mlp_sin
        + variables_meteorologicas
    )

    # Todas las entradas requeridas por los modelos.
    columnas_caracteristicas = list(
        dict.fromkeys(
            features_mlp_con
            + features_fuente
        )
    )

    # Máscara común para comparar pronósticos.
    # El origen y el objetivo deben pertenecer a la misma partición.
    # Common forecast mask: origin and target must share a partition.
    # The intermediate observation C(t+1) is not required for h > 1.
    mascara_pronostico = (
        muestra["valido"]
        & muestra["C_lag24"].notna()
        & muestra["C_objetivo"].notna()
        & (
            muestra["particion"]
            == muestra["particion_objetivo"]
        )
    )

    # Exigir entradas completas y finitas.
    matriz_caracteristicas = muestra[
        columnas_caracteristicas
    ].to_numpy(dtype=float)

    mascara_entradas = (
        muestra[columnas_caracteristicas]
        .notna()
        .all(axis=1)
        .to_numpy()
        & np.isfinite(matriz_caracteristicas).all(axis=1)
    )

    mascara_pronostico &= mascara_entradas

    ejemplos = muestra.loc[
        mascara_pronostico
    ].copy()

    if ejemplos.empty:
        raise ValueError(
            f"No hay ejemplos vÃ¡lidos para horizonte={horizonte}."
        )

    train = ejemplos[
        ejemplos["particion"] == "train"
    ].copy()

    val = ejemplos[
        ejemplos["particion"] == "validation"
    ].copy()

    test = ejemplos[
        ejemplos["particion"] == "test"
    ].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(
            "Alguna particiÃ³n quedÃ³ vacÃ­a. "
            "Comprueba los nombres de particiÃ³n del CSV."
        )

    print(
        f"Ejemplos comunes: train={len(train)}, "
        f"validation={len(val)}, test={len(test)}"
    )

    # --------------------------------------------------------
    # 6.1 Ajustar escaladores solo con train
    # --------------------------------------------------------

    definicion_features = {
        "sin": features_mlp_sin,
        "con": features_mlp_con,
        "fuente": features_fuente,
    }

    escaladores = {}
    matrices = {}

    for nombre, columnas in definicion_features.items():
        escalador = StandardScaler()

        # Ãšnicamente train determina medias y desviaciones estÃ¡ndar.
        escalador.fit(train[columnas])

        escaladores[nombre] = escalador
        matrices[nombre] = {}

        for particion, subconjunto in [
            ("train", train),
            ("validation", val),
            ("test", test),
        ]:
            matrices[nombre][particion] = convertir_tensor(
                escalador.transform(
                    subconjunto[columnas]
                )
            )

    # Concentraciones en sus unidades originales de Âµg/mÂ³.
    concentraciones = {}

    for particion, subconjunto in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:
        concentraciones[particion] = {
            "actual": convertir_tensor(
                subconjunto["C_actual"].to_numpy()
            ),
            "objetivo": convertir_tensor(
                subconjunto["C_objetivo"].to_numpy()
            ),
            "siguiente": convertir_tensor(
                subconjunto["C_siguiente"].to_numpy()
            ),
        }

    # --------------------------------------------------------
    # 6.2 Preparar transiciones fÃ­sicas vÃ¡lidas
    # --------------------------------------------------------

    # Para calcular el residuo de una hora no es obligatorio que
    # todos los datos del horizonte h sean vÃ¡lidos. Se necesitan
    # Ãºnicamente C(t), C(t+1), entradas en t y una misma particiÃ³n.
    #
    # La pÃ©rdida fÃ­sica usa transiciones dentro de cada particiÃ³n;
    # no utiliza transiciones que atraviesen una frontera.

    muestra["C_siguiente_fisica"] = muestra["C"].shift(-1)
    muestra["particion_fisica_siguiente"] = (
        muestra["particion"].shift(-1)
    )

    mascara_fisica = (
        muestra["valido"]
        & muestra["C_siguiente_fisica"].notna()
        & (
            muestra["particion"]
            == muestra["particion_fisica_siguiente"]
        )
    )

    # La fuente se evalÃºa con caracterÃ­sticas del origen t.
    mascara_fisica &= (
        muestra[features_fuente]
        .notna()
        .all(axis=1)
    )

    valores_fuente = muestra[
        features_fuente
    ].to_numpy(dtype=float)

    mascara_fisica &= np.isfinite(
        valores_fuente
    ).all(axis=1)

    transiciones = muestra.loc[
        mascara_fisica
    ].copy()

    trans_train = transiciones[
        transiciones["particion"] == "train"
    ].copy()

    trans_val = transiciones[
        transiciones["particion"] == "validation"
    ].copy()

    # El residuo fÃ­sico se evalÃºa en transiciones reales y vÃ¡lidas
    # de train y validation. No se usa test para entrenar ni validar.
    if len(trans_train) == 0 or len(trans_val) == 0:
        raise ValueError(
            "No hay transiciones fÃ­sicas suficientes en train "
            "o validation."
        )

    x_residuo_train = convertir_tensor(
        escaladores["fuente"].transform(
            trans_train[features_fuente]
        )
    )

    c_residuo_train = convertir_tensor(
        trans_train["C"].to_numpy()
    )

    c_siguiente_train = convertir_tensor(
        trans_train["C_siguiente_fisica"].to_numpy()
    )

    x_residuo_val = convertir_tensor(
        escaladores["fuente"].transform(
            trans_val[features_fuente]
        )
    )

    c_residuo_val = convertir_tensor(
        trans_val["C"].to_numpy()
    )

    c_siguiente_val = convertir_tensor(
        trans_val["C_siguiente_fisica"].to_numpy()
    )

    print(
        f"Transiciones fÃ­sicas: train={len(trans_train)}, "
        f"validation={len(trans_val)}"
    )

    # --------------------------------------------------------
    # 6.3 Entrenar MLP sin meteorologÃ­a
    # --------------------------------------------------------

    modelo_sin, epoca_sin, perdida_sin = entrenar_mlp(
        matrices["sin"]["train"],
        concentraciones["train"]["objetivo"],
        matrices["sin"]["validation"],
        concentraciones["validation"]["objetivo"],
    )

    print(
        f"MLP sin meteorologÃ­a: mejor Ã©poca={epoca_sin}; "
        f"MSE validaciÃ³n={perdida_sin:.4f}"
    )

    # --------------------------------------------------------
    # 6.4 Entrenar MLP con meteorologÃ­a
    # --------------------------------------------------------

    modelo_con, epoca_con, perdida_con = entrenar_mlp(
        matrices["con"]["train"],
        concentraciones["train"]["objetivo"],
        matrices["con"]["validation"],
        concentraciones["validation"]["objetivo"],
    )

    print(
        f"MLP con meteorologÃ­a: mejor Ã©poca={epoca_con}; "
        f"MSE validaciÃ³n={perdida_con:.4f}"
    )

    # --------------------------------------------------------
    # 6.5 Entrenar el modelo neuronal fÃ­sico
    # --------------------------------------------------------

    modelo_fisico, epoca_fisica, perdida_fisica = (
        entrenar_modelo_fisico(
            x_train=matrices["fuente"]["train"],
            c_train=concentraciones["train"]["actual"],
            y_train=concentraciones["train"]["objetivo"],
            x_val=matrices["fuente"]["validation"],
            c_val=concentraciones["validation"]["actual"],
            y_val=concentraciones["validation"]["objetivo"],
            x_residuo_train=x_residuo_train,
            c_residuo_train=c_residuo_train,
            c_siguiente_train=c_siguiente_train,
            x_residuo_val=x_residuo_val,
            c_residuo_val=c_residuo_val,
            c_siguiente_val=c_siguiente_val,
            horizonte=horizonte,
        )
    )

    modelo_fisico.eval()

    with torch.no_grad():
        k_aprendido = float(
            modelo_fisico.obtener_k().item()
        )

    print(
        f"Modelo neuronal fÃ­sico: mejor Ã©poca={epoca_fisica}; "
        f"MSE validaciÃ³n combinada={perdida_fisica:.4f}; "
        f"k={k_aprendido:.6f} hâ»Â¹"
    )

    # Registrar las Ã©pocas seleccionadas.
    resumen_entrenamiento.append({
        "horizonte_horas": horizonte,
        "modelo": "MLP_sin_meteorologia",
        "mejor_epoca": epoca_sin,
        "perdida_validacion": perdida_sin,
    })

    resumen_entrenamiento.append({
        "horizonte_horas": horizonte,
        "modelo": "MLP_con_meteorologia",
        "mejor_epoca": epoca_con,
        "perdida_validacion": perdida_con,
    })

    resumen_entrenamiento.append({
        "horizonte_horas": horizonte,
        "modelo": "Modelo_neuronal_fisico",
        "mejor_epoca": epoca_fisica,
        "perdida_validacion": perdida_fisica,
    })

    # --------------------------------------------------------
    # 6.6 Evaluar validaciÃ³n y test
    # --------------------------------------------------------

    for particion in ["validation", "test"]:
        c_actual = concentraciones[particion]["actual"]
        y_real = concentraciones[particion]["objetivo"]

        # Persistencia: pronostica que el valor futuro serÃ¡ igual
        # al Ãºltimo valor observado en el origen del pronÃ³stico.
        predicciones = {
            "Persistencia": c_actual.cpu().numpy()
        }

        # MLP sin meteorologÃ­a.
        modelo_sin.eval()

        with torch.no_grad():
            predicciones["MLP_sin_meteorologia"] = (
                modelo_sin(
                    matrices["sin"][particion]
                )
                .cpu()
                .numpy()
            )

        # MLP con meteorologÃ­a del origen.
        modelo_con.eval()

        with torch.no_grad():
            predicciones["MLP_con_meteorologia"] = (
                modelo_con(
                    matrices["con"][particion]
                )
                .cpu()
                .numpy()
            )

        # Modelo con ecuaciÃ³n dinÃ¡mica.
        with torch.no_grad():
            predicciones["Modelo_neuronal_fisico"] = (
                modelo_fisico.predecir(
                    matrices["fuente"][particion],
                    c_actual,
                    horizonte,
                )
                .cpu()
                .numpy()
            )

        y_numpy = y_real.cpu().numpy()

        # El tensor de evaluación se construyó a partir de val o test,
        # que conservan el mismo orden de filas que los tensores.
        subconjunto_evaluacion = val if particion == "validation" else test

        # fecha_utc representa el origen t; el objetivo se encuentra en t+h.
        fechas_objetivo = (
            subconjunto_evaluacion["fecha_utc"]
            + pd.to_timedelta(horizonte, unit="h")
        ).reset_index(drop=True)

        # Validaciones explícitas para impedir una asociación incorrecta
        # entre predicciones y fechas.
        if len(fechas_objetivo) != len(y_numpy):
            raise ValueError(
                f"Desalineación temporal: horizonte={horizonte}, "
                f"partición={particion}, fechas={len(fechas_objetivo)}, "
                f"objetivos={len(y_numpy)}."
            )

        for nombre_modelo, prediccion in predicciones.items():
            if len(prediccion) != len(y_numpy):
                raise ValueError(
                    f"Desalineación en {nombre_modelo}: "
                    f"horizonte={horizonte}, partición={particion}."
                )

            # Guardar el valor real, el pronóstico y los errores por instante.
            for fecha, real, estimado in zip(
                fechas_objetivo,
                y_numpy,
                prediccion,
            ):
                predicciones_detalladas.append({
                    "fecha_objetivo": fecha,
                    "horizonte_horas": horizonte,
                    "particion": particion,
                    "modelo": nombre_modelo,
                    "o3_real": float(real),
                    "o3_predicho": float(estimado),
                    # Residuo definido como observado menos predicho.
                    "residuo": float(real - estimado),
                    "error_absoluto": float(abs(real - estimado)),
                    "error_cuadratico": float((real - estimado) ** 2),
                })

            metricas = calcular_metricas(
                y_numpy,
                prediccion,
            )

            resultados.append({
                "horizonte_horas": horizonte,
                "particion": particion,
                "modelo": nombre_modelo,
                **metricas,
                "mejor_epoca": {
                    "MLP_sin_meteorologia": epoca_sin,
                    "MLP_con_meteorologia": epoca_con,
                    "Modelo_neuronal_fisico": epoca_fisica,
                    "Persistencia": 0,
                }[nombre_modelo],
                "k_por_hora": (
                    k_aprendido
                    if nombre_modelo == "Modelo_neuronal_fisico"
                    else np.nan
                ),
                "peso_fisico": (
                    PESO_FISICO
                    if nombre_modelo == "Modelo_neuronal_fisico"
                    else np.nan
                ),
            })


# ------------------------------------------------------------
# 7. Guardar resultados
# ------------------------------------------------------------

tabla_resultados = pd.DataFrame(resultados)

ruta_metricas = (
    RUTA_SALIDA
    / "o3_pinn_comparacion_metricas.csv"
)

tabla_resultados.to_csv(
    ruta_metricas,
    index=False,
)

tabla_entrenamiento = pd.DataFrame(
    resumen_entrenamiento
)

ruta_entrenamiento = (
    RUTA_SALIDA
    / "o3_pinn_resumen_entrenamiento.csv"
)

tabla_entrenamiento.to_csv(
    ruta_entrenamiento,
    index=False,
)

# Exportar predicciones individuales para análisis de residuos y bootstrap.
tabla_predicciones = pd.DataFrame(predicciones_detalladas)

ruta_predicciones = (
    RUTA_SALIDA
    / "o3_pinn_predicciones_detalladas.csv"
)

tabla_predicciones.to_csv(
    ruta_predicciones,
    index=False,
    date_format="%Y-%m-%dT%H:%M:%S",
)

print("\n" + "=" * 80)
print("RESULTADOS DE VALIDACIÃ“N Y TEST")
print("=" * 80)

print(
    tabla_resultados.sort_values(
        ["horizonte_horas", "particion", "MAE"]
    ).to_string(index=False)
)

print("\nRESUMEN DEL ENTRENAMIENTO")
print(
    tabla_entrenamiento.to_string(index=False)
)

print(f"\nMÃ©tricas guardadas en:\n{ruta_metricas}")
print(f"Entrenamiento guardado en:\n{ruta_entrenamiento}")

print(f"Predicciones detalladas guardadas en: {ruta_predicciones}")
print(f"Filas de predicciones guardadas: {len(tabla_predicciones)}")
