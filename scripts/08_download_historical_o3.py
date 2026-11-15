
# scripts/08_download_historical_o3.py
# Descarga histórica mensual de O3 de Londres desde OpenAQ v3.
# Conserva cada respuesta JSON original y permite reanudar la descarga.
# Detiene el proceso si OpenAQ devuelve HTTP 429 (límite de solicitudes).

import os
import json
import time
import calendar
from pathlib import Path
from datetime import date
import requests
from dotenv import load_dotenv

# Identifica la raíz del proyecto y carga las credenciales.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENAQ_API_KEY")

# Sensor de O3 de Londres que ya validamos preliminarmente.
SENSOR_ID = 1304560
URL = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/hours"

# Periodo histórico que queremos estudiar.
FECHA_INICIO = date(2021, 4, 1)
FECHA_FIN_EXCLUSIVA = date(2024, 1, 1)

# Carpeta de salida: un archivo JSON por mes.
CARPETA_SALIDA = ROOT / "data" / "raw" / "openaq_historical_o3"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# Pausa entre consultas para reducir la presión sobre la API.
PAUSA_SEGUNDOS = 3


def meses_del_periodo(inicio, fin_exclusivo):
    """Genera pares de fechas para cada mes del periodo solicitado."""
    anio = inicio.year
    mes = inicio.month

    while date(anio, mes, 1) < fin_exclusivo:
        primer_dia = date(anio, mes, 1)

        # Calcula el primer día del mes siguiente.
        if mes == 12:
            siguiente_mes = date(anio + 1, 1, 1)
        else:
            siguiente_mes = date(anio, mes + 1, 1)

        # Evita solicitar fechas anteriores al inicio elegido.
        desde = max(primer_dia, inicio)
        hasta = min(siguiente_mes, fin_exclusivo)

        yield desde, hasta

        anio, mes = siguiente_mes.year, siguiente_mes.month


def main():
    """Descarga cada mes y guarda la respuesta sin modificarla."""
    if not API_KEY:
        print("ERROR: no se encontró OPENAQ_API_KEY en el archivo .env.")
        return

    # La clave se envía en el encabezado, no en la URL.
    encabezados = {"X-API-Key": API_KEY}
    meses = list(meses_del_periodo(FECHA_INICIO, FECHA_FIN_EXCLUSIVA))

    print(f"Sensor: {SENSOR_ID}")
    print(f"Meses planificados: {len(meses)}")
    print(f"Carpeta de salida: {CARPETA_SALIDA}")
    print("No se sobrescribirán los archivos existentes.\n")

    for indice, (desde, hasta) in enumerate(meses, start=1):
        nombre = f"london_o3_sensor_{SENSOR_ID}_{desde:%Y-%m}.json"
        ruta = CARPETA_SALIDA / nombre

        # Si el archivo ya existe, se conserva y se continúa.
        if ruta.exists():
            print(f"[{indice}/{len(meses)}] Ya existe: {nombre}")
            continue

        parametros = {
            "datetime_from": f"{desde.isoformat()}T00:00:00Z",
            "datetime_to": f"{hasta.isoformat()}T00:00:00Z",
            "limit": 1000,
            "page": 1,
        }

        print(
            f"[{indice}/{len(meses)}] Descargando "
            f"{desde:%Y-%m} ..."
        )

        try:
            respuesta = requests.get(
                URL,
                headers=encabezados,
                params=parametros,
                timeout=60,
            )

            # Si se supera el límite, detenerse sin seguir consultando.
            if respuesta.status_code == 429:
                print("HTTP 429: límite de solicitudes alcanzado.")
                print("Detén el proceso y espera antes de reanudar.")
                return

            respuesta.raise_for_status()
            datos = respuesta.json()

            # Verifica que la respuesta tenga la estructura esperada.
            if not isinstance(datos, dict) or not isinstance(
                datos.get("results"), list
            ):
                print("Respuesta inesperada; no se guardará este mes.")
                print(str(datos)[:500])
                return

            # Comprueba si la API informa más registros de los devueltos.
            meta = datos.get("meta", {})
            encontrados = meta.get("found")
            recibidos = len(datos["results"])

            if isinstance(encontrados, int) and encontrados > recibidos:
                print(
                    f"AVISO: la API indica {encontrados} registros, "
                    f"pero devolvió {recibidos}. "
                    "No se guardará una respuesta incompleta."
                )
                return

            # Guarda la respuesta original para conservar su trazabilidad.
            with ruta.open("w", encoding="utf-8") as archivo:
                json.dump(datos, archivo, ensure_ascii=False, indent=2)

            print(
                f"Guardado: {nombre} | "
                f"registros: {recibidos} | "
                f"reportados: {encontrados}"
            )

        except requests.RequestException as error:
            print(f"Error de conexión/API en {desde:%Y-%m}: {error}")
            print("Se detiene la descarga para evitar errores repetidos.")
            return
        except (ValueError, OSError) as error:
            print(f"Error procesando {desde:%Y-%m}: {error}")
            return

        # Pausa entre peticiones para reducir la frecuencia de consultas.
        time.sleep(PAUSA_SEGUNDOS)

    print("\nDescarga mensual terminada.")
    print("El siguiente paso será auditar cobertura, fechas y calidad.")


if __name__ == "__main__":
    main()
