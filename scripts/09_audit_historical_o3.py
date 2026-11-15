
# scripts/09_audit_historical_o3.py
# Audita los archivos mensuales de O3 descargados desde OpenAQ.
# Genera una tabla de calidad mensual y un CSV con las observaciones.
# No modifica los archivos JSON originales.

import json
import calendar
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# 1. Configuración de rutas
# --------------------------------------------------

SENSOR_ID = 1304560

CARPETA_ENTRADA = Path("data/raw/openaq_historical_o3")
CARPETA_SALIDA = Path("outputs/tables")

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVO_AUDITORIA = CARPETA_SALIDA / "o3_auditoria_mensual.csv"
ARCHIVO_SERIE = CARPETA_SALIDA / "o3_serie_consolidada.csv"


# --------------------------------------------------
# 2. Funciones auxiliares
# --------------------------------------------------

def obtener_fecha(registro):
    """Extrae la fecha UTC usando las estructuras habituales de OpenAQ."""

    # Estructura habitual de las mediciones por periodo.
    periodo = registro.get("period", {})
    if isinstance(periodo, dict):
        fecha = periodo.get("datetimeFrom", {})
        if isinstance(fecha, dict) and fecha.get("utc"):
            return fecha["utc"]

    # Estructuras alternativas, por si cambia el formato.
    fecha = registro.get("datetime", {})
    if isinstance(fecha, dict) and fecha.get("utc"):
        return fecha["utc"]

    if isinstance(fecha, str):
        return fecha

    return None


def tiene_bandera(registro):
    """Detecta banderas de calidad sin confundirlas con un campo vacío."""

    bandera = registro.get("flagInfo")

    # Una bandera puede venir como diccionario, lista o valor booleano.
    if isinstance(bandera, dict):
        return bool(bandera)

    if isinstance(bandera, list):
        return len(bandera) > 0

    if isinstance(bandera, bool):
        return bandera

    return False


def obtener_unidad(registro):
    """Busca la unidad de medida en los campos disponibles."""

    # OpenAQ puede incluir las unidades dentro de parameter.
    parametro = registro.get("parameter", {})
    if isinstance(parametro, dict) and parametro.get("units"):
        return parametro["units"]

    # Algunas respuestas pueden incluir unidades en otra estructura.
    sensor = registro.get("sensor", {})
    if isinstance(sensor, dict):
        parametro_sensor = sensor.get("parameter", {})
        if isinstance(parametro_sensor, dict):
            return parametro_sensor.get("units")

    return None


# --------------------------------------------------
# 3. Buscar y procesar los archivos mensuales
# --------------------------------------------------

archivos = sorted(
    CARPETA_ENTRADA.glob(f"london_o3_sensor_{SENSOR_ID}_*.json")
)

if not archivos:
    raise FileNotFoundError(
        f"No se encontraron archivos en {CARPETA_ENTRADA.resolve()}"
    )

observaciones = []
resumen_mensual = []
errores = []

for archivo in archivos:
    print(f"Auditando: {archivo.name}")

    try:
        # Abrir la respuesta original de la API.
        with archivo.open("r", encoding="utf-8") as f:
            datos = json.load(f)

        resultados = datos.get("results", [])

        if not isinstance(resultados, list):
            errores.append(f"{archivo.name}: 'results' no es una lista")
            continue

        # El nombre del archivo permite identificar el mes consultado.
        mes = archivo.stem.split("_")[-1]

        # Calcular las horas de ese mes para estimar la cobertura.
        anio, numero_mes = map(int, mes.split("-"))
        dias_mes = calendar.monthrange(anio, numero_mes)[1]
        horas_esperadas = dias_mes * 24

        fechas = []
        valores = []
        unidades = set()
        registros_con_bandera = 0
        fechas_invalidas = 0

        # Extraer las observaciones sin eliminar los registros originales.
        for registro in resultados:
            fecha = obtener_fecha(registro)
            valor = registro.get("value")

            if fecha is None:
                fechas_invalidas += 1
                continue

            # Convertir la fecha a UTC.
            fecha_utc = pd.to_datetime(fecha, utc=True, errors="coerce")

            if pd.isna(fecha_utc):
                fechas_invalidas += 1
                continue

            # Convertir el valor a número; los valores no válidos serán nulos.
            valor_numerico = pd.to_numeric(valor, errors="coerce")

            if pd.notna(valor_numerico):
                valor_numerico = float(valor_numerico)

            unidad = obtener_unidad(registro)
            if unidad:
                unidades.add(str(unidad))

            marcada = tiene_bandera(registro)
            if marcada:
                registros_con_bandera += 1

            fechas.append(fecha_utc)
            valores.append(valor_numerico)

            observaciones.append({
                "fecha_utc": fecha_utc,
                "valor_o3": valor_numerico,
                "unidad": unidad,
                "tiene_bandera": marcada,
                "archivo_origen": archivo.name
            })

        # Crear una tabla temporal para revisar duplicados y continuidad.
        temporal = pd.DataFrame({
            "fecha_utc": fechas,
            "valor_o3": valores
        })

        if not temporal.empty:
            duplicados = int(
                temporal["fecha_utc"].duplicated(keep=False).sum()
            )

            fechas_unicas = temporal["fecha_utc"].drop_duplicates().sort_values()

            # Las diferencias permiten localizar interrupciones temporales.
            intervalos = fechas_unicas.diff().dropna()
            intervalos_largos = int((intervalos > pd.Timedelta(hours=1.5)).sum())

            primera_fecha = fechas_unicas.min()
            ultima_fecha = fechas_unicas.max()

            valores_validos = temporal["valor_o3"].dropna()

            minimo = valores_validos.min() if not valores_validos.empty else None
            maximo = valores_validos.max() if not valores_validos.empty else None
            promedio = valores_validos.mean() if not valores_validos.empty else None

        else:
            duplicados = 0
            intervalos_largos = 0
            primera_fecha = None
            ultima_fecha = None
            minimo = maximo = promedio = None

        total_registros = len(resultados)
        valores_nulos = int(pd.isna(pd.Series(valores)).sum())

        # Esta cobertura usa el mes calendario completo como referencia.
        # Si el sensor comenzó o terminó a mitad de mes, será baja por diseño.
        cobertura = (
            100 * temporal["fecha_utc"].nunique() / horas_esperadas
            if not temporal.empty else 0.0
        )

        resumen_mensual.append({
            "mes": mes,
            "registros_api": total_registros,
            "fechas_validas": len(temporal),
            "fechas_unicas": (
                int(temporal["fecha_utc"].nunique())
                if not temporal.empty else 0
            ),
            "valores_nulos": valores_nulos,
            "registros_con_bandera": registros_con_bandera,
            "fechas_invalidas": fechas_invalidas,
            "duplicados_temporales": duplicados,
            "intervalos_mayores_1_5h": intervalos_largos,
            "horas_esperadas_mes": horas_esperadas,
            "cobertura_mensual_pct": round(cobertura, 2),
            "primera_fecha_utc": primera_fecha,
            "ultima_fecha_utc": ultima_fecha,
            "unidad_detectada": "; ".join(sorted(unidades)),
            "minimo": minimo,
            "maximo": maximo,
            "promedio": promedio
        })

    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        errores.append(f"{archivo.name}: {error}")


# --------------------------------------------------
# 4. Guardar los resultados de la auditoría
# --------------------------------------------------

auditoria = pd.DataFrame(resumen_mensual)

if not auditoria.empty:
    auditoria = auditoria.sort_values("mes")
    auditoria.to_csv(ARCHIVO_AUDITORIA, index=False, encoding="utf-8-sig")

if observaciones:
    serie = pd.DataFrame(observaciones)

    # Ordenar cronológicamente y conservar los duplicados para poder auditarlos.
    serie = serie.sort_values("fecha_utc")

    serie.to_csv(ARCHIVO_SERIE, index=False, encoding="utf-8-sig")

# --------------------------------------------------
# 5. Mostrar un resumen legible en la terminal
# --------------------------------------------------

print("\n" + "=" * 70)
print("RESUMEN DE AUDITORÍA DE O3")
print("=" * 70)

print(f"Archivos encontrados: {len(archivos)}")
print(f"Meses auditados: {len(auditoria)}")
print(f"Observaciones extraídas: {len(observaciones)}")

if not auditoria.empty:
    columnas_resumen = [
        "mes",
        "registros_api",
        "valores_nulos",
        "registros_con_bandera",
        "duplicados_temporales",
        "cobertura_mensual_pct"
    ]

    print("\nResumen mensual:")
    print(auditoria[columnas_resumen].to_string(index=False))

    print("\nUnidades detectadas:")
    print(auditoria["unidad_detectada"].value_counts(dropna=False).to_string())

    print("\nCobertura mensual más baja:")
    print(
        auditoria.nsmallest(5, "cobertura_mensual_pct")[
            ["mes", "cobertura_mensual_pct", "registros_api"]
        ].to_string(index=False)
    )

    print("\nCobertura mensual más alta:")
    print(
        auditoria.nlargest(5, "cobertura_mensual_pct")[
            ["mes", "cobertura_mensual_pct", "registros_api"]
        ].to_string(index=False)
    )

print("\nArchivos generados:")
print(f"- {ARCHIVO_AUDITORIA}")
print(f"- {ARCHIVO_SERIE}")

if errores:
    print("\nAdvertencias o errores:")
    for error in errores:
        print(f"- {error}")
else:
    print("\nNo se detectaron errores de lectura en los archivos.")
