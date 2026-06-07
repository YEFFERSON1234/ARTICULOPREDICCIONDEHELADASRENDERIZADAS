import xarray as xr
import pandas as pd
import glob
import os

# 1. Detectar automáticamente las rutas relativas
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Carpeta de origen donde tu script guardó los recortes de MERRA-2
carpeta_origen = os.path.join(directorio_actual, 'data_merra2_puno')

# Nueva carpeta de destino para los archivos .csv resultantes
carpeta_destino = os.path.join(directorio_actual, 'data_merra2_puno_csv')

# Crear la carpeta de destino si aún no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)
    print(f"-> Se ha creado la carpeta de destino: {carpeta_destino}")

# 2. Buscar todos los archivos .nc que sigan el patrón 'merra2_puno_*.nc'
archivos_nc = glob.glob(os.path.join(carpeta_origen, "merra2_puno_*.nc"))
archivos_nc.sort()

if not archivos_nc:
    print(f"[!] No se encontraron archivos .nc en la carpeta: {carpeta_origen}")
    print("Asegúrate de haber ejecutado primero tu script de descarga.")
    exit()

print(f"Se encontraron {len(archivos_nc)} archivos diarios de MERRA-2 para convertir.\n" + "-"*50)

# 3. Bucle de conversión archivo por archivo (Evita el consumo excesivo de RAM)
for index, ruta_nc in enumerate(archivos_nc):
    nombre_base = os.path.basename(ruta_nc)  # Ej: 'merra2_puno_20230201.nc'
    nombre_csv = nombre_base.replace('.nc', '.csv')  # Ej: 'merra2_puno_20230201.csv'
    ruta_final_csv = os.path.join(carpeta_destino, nombre_csv)
    
    print(f"[{index + 1}/{len(archivos_nc)}] Procesando: {nombre_base}...")
    
    try:
        # Abrir el archivo de forma segura con un manejador de contexto
        with xr.open_dataset(ruta_nc) as ds:
            # Convertir el cubo xarray a una estructura de tabla normal (Dataframe)
            # Esto convierte automáticamente las dimensiones (time, lat, lon) en columnas
            df = ds.to_dataframe().reset_index()
            
            # NOTA DE SEGURIDAD: Como ya restaste 273.15 en tu script de descarga,
            # aquí los datos numéricos de T2M y TS se transfieren idénticos en Celsius.
            
            # Guardar el DataFrame como un archivo CSV individual
            df.to_csv(ruta_final_csv, index=False)
            print(f"   [OK] Convertido y guardado en: data_merra2_puno_csv/{nombre_csv}")
            
    except Exception as e:
        print(f"   [!] Error al convertir el archivo {nombre_base}: {e}")
        import traceback
        traceback.print_exc()

print("-"*50 + "\n¡Proceso finalizado! Todos los días de MERRA-2 están migrados a formato CSV.")