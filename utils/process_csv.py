import os
import glob
import xarray as xr
import pandas as pd
import warnings

# Ignorar advertencias molestas en la consola
warnings.filterwarnings('ignore')

def procesar_nc_a_csv(ruta_entrada: str, ruta_salida: str):
    print("="*60)
    print("INICIANDO CONVERSION OPTIMIZADA DE ERA5 (.nc a .csv)")
    print("="*60)

    # 1. Buscar archivos
    archivos_nc = glob.glob(os.path.join(ruta_entrada, '*.nc'))
    
    if not archivos_nc:
        print(f"[ALERTA] Carpeta vacia! No se encontraron archivos .nc en: {ruta_entrada}")
        return

    print(f"[OK] Encontrados {len(archivos_nc)} archivos. Procesando...\n")
    lista_dataframes = []

    # Variables que nos interesan de ERA5
    vars_objetivo = ['t2m', 'tp', 'sp', 'd2m']
    renombres_clima = {
        't2m': 'temp_2m_era5',
        'tp': 'precip_era5',
        'sp': 'presion_era5',
        'd2m': 'dew_point_era5'
    }

    # 2. Procesar cada archivo de forma eficiente
    for archivo in archivos_nc:
        try:
            ds = xr.open_dataset(archivo)
            
            # Filtrar variables
            vars_presentes = [var for var in vars_objetivo if var in ds.variables]
            
            # Si el archivo no tiene nuestras variables, lo saltamos
            if not vars_presentes:
                ds.close()
                continue
                
            ds_filtrado = ds[vars_presentes]
            df = ds_filtrado.to_dataframe().reset_index()
            
            # Buscamos el nombre real de la columna de tiempo y lo forzamos a "fecha"
            nombres_tiempo = ['time', 'valid_time', 'Time', 'date', 'step']
            for col in nombres_tiempo:
                if col in df.columns:
                    df = df.rename(columns={col: 'fecha'})
                    break # Si ya encontró uno, deja de buscar
            
            # Asegurarnos de que latitud y longitud estén bien escritas
            if 'lat' in df.columns: df = df.rename(columns={'lat': 'latitude'})
            if 'lon' in df.columns: df = df.rename(columns={'lon': 'longitude'})

            # Renombrar columnas de clima
            df = df.rename(columns=renombres_clima)
            
            # Convertir de Kelvin a Celsius
            if 'temp_2m_era5' in df.columns:
                df['temp_2m_era5'] -= 273.15
            if 'dew_point_era5' in df.columns:
                df['dew_point_era5'] -= 273.15

            lista_dataframes.append(df)
            print(f"  [OK] Procesado: {os.path.basename(archivo)}")
            
            ds.close()
            
        except Exception as e:
            print(f"  [ERROR] Fallo en {os.path.basename(archivo)}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Unificar y limpiar
    if not lista_dataframes:
        print("\n[ALERTA] No se extrajo ningun dato valido.")
        return

    print("\nUniendo datos y calculando promedios diarios...")
    df_final = pd.concat(lista_dataframes, ignore_index=True)
    
    # Comprobación de seguridad final
    if 'fecha' not in df_final.columns:
        print("\n[ERROR CRITICO] Ningun archivo tenia una columna de tiempo reconocible.")
        return

    # Quitar horas de la fecha para que coincida con SENAMHI (diario)
    df_final['fecha'] = pd.to_datetime(df_final['fecha']).dt.date
    
    # === LA SOLUCIÓN ESTÁ AQUÍ (numeric_only=True) ===
    # Agrupar por dia y coordenadas, promediando SOLO las columnas numéricas
    df_final = df_final.groupby(['fecha', 'latitude', 'longitude']).mean(numeric_only=True).reset_index()

    # Redondeamos a 2 decimales
    df_final['latitude'] = df_final['latitude'].round(2)
    df_final['longitude'] = df_final['longitude'].round(2)

    # 4. Guardar resultado
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df_final.to_csv(ruta_salida, index=False)
    
    print("\n" + "="*60)
    print(f"[EXITO] Archivo CSV Maestro guardado en: {ruta_salida}")
    print(f"Total de filas generadas: {len(df_final)}")
    print("="*60)
    print(df_final.head())

# ==========================================
# EJECUCION
# ==========================================
if __name__ == '__main__':
    CARPETA_NC = 'data/datos_era5_puno' 
    ARCHIVO_CSV_SALIDA = 'data/era5_procesado_maestro.csv'
    procesar_nc_a_csv(CARPETA_NC, ARCHIVO_CSV_SALIDA)