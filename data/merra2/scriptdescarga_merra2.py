import os
import sys

# 1. Configuración de Credenciales de NASA Earthdata
# Credentials must be set via environment variables or a .env file.
# See .env.example for the required variable names.
if not os.environ.get("EARTHDATA_USERNAME") or not os.environ.get("EARTHDATA_PASSWORD"):
    print("ERROR: Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.")
    print("See .env.example for details.")
    sys.exit(1)

import earthaccess

import xarray as xr
import pandas as pd
import requests

# 2. Autenticación automática mediante entorno
print("Autenticando con NASA Earthdata...")
auth = earthaccess.login(strategy="environment")

# Bounding Box para la región de Puno [Oeste, Sur, Este, Norte]
puno_bbox = (-71.1, -17.5, -68.8, -13.0)

# Rango temporal de ejemplo (puedes ampliarlo según tus necesidades de entrenamiento)
fecha_inicio = "2023-03-23"
fecha_fin = "2023-07-31"

print("Buscando gránulos en el catálogo de la NASA...")
# M2T1NXSLV: Dataset por hora de variables en superficie de MERRA-2
results = earthaccess.search_data(
    short_name="M2T1NXSLV",
    temporal=(fecha_inicio, fecha_fin),
    bounding_box=puno_bbox
)

print(f"Se encontraron {len(results)} archivos diarios disponibles.")

# Crear carpeta para almacenar los recortes de Puno
output_dir = "./data_merra2_puno"
os.makedirs(output_dir, exist_ok=True)

# 3. Obtención de enlaces HTTP directos
print("Obteniendo enlaces de descarga...")
download_urls = [granule.data_links(access="external")[0] for granule in results]

# Crear sesión de descarga con las credenciales inyectadas
session = earthaccess.get_requests_https_session()

# 4. Bucle robusto de descarga, recorte espacial y conversión
for i, url in enumerate(download_urls):
    temp_file = f"temp_global_{i}.nc"
    try:
        print(f"\n[{i+1}/{len(download_urls)}] Descargando archivo temporal de la NASA...")
        
        # Descarga el archivo global por streaming en pequeños bloques
        with session.get(url, stream=True) as r:
            r.raise_for_status()
            with open(temp_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Procesamiento del archivo con xarray
        with xr.open_dataset(temp_file) as ds:
            # Variables críticas para predecir heladas por radiación/advección
            variables_interes = ['T2M', 'TS', 'QV2M', 'U2M', 'V2M']
            
            # Recorte exacto para las coordenadas de Puno
            ds_puno = ds[variables_interes].sel(
                lat=slice(-17.5, -13.0),
                lon=slice(-71.1, -68.8)
            )
            
            # Conversión de Kelvin a Celsius para las temperaturas
            ds_puno['T2M'] = ds_puno['T2M'] - 273.15
            ds_puno['TS'] = ds_puno['TS'] - 273.15
            
            # Formatear la fecha para nombrar el archivo final de forma ordenada
            fecha_str = pd.to_datetime(ds_puno.time.values[0]).strftime('%Y%m%d')
            file_name = f"{output_dir}/merra2_puno_{fecha_str}.nc"
            
            # Guardar el dataset localmente (pesará solo unos kilobytes)
            ds_puno.to_netcdf(file_name)
            print(f"-> ¡Éxito! Guardado en: {file_name}")
            
    except Exception as e:
        print(f"-> Error procesando el archivo {i+1}: {e}")
        
    finally:
        # Eliminación estricta del archivo global temporal para no saturar el almacenamiento
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except PermissionError:
                pass 

print("\n¡Proceso finalizado con éxito! Los datos optimizados de Puno están listos para tu modelo.")