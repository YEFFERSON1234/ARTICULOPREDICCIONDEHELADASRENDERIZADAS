"""
DESCARGADOR Y PROCESADOR MODIS CORREGIDO
Proyecto: Predicción de Heladas con IA + OpenGL
"""

import os
import sys
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# =========================================================
# ⚠️ CONFIGURACIÓN IMPORTANTE ⚠️
# =========================================================
# PON AQUÍ EL ID DE TU PROYECTO DE GOOGLE CLOUD:
EE_PROJECT_ID = "PON_AQUI_TU_ID_DE_PROYECTO" # <-- ¡CÁMBIALO AQUÍ!

# Región Puno (Coordenadas ajustadas)
LAT_MIN = -17.5
LAT_MAX = -14.5
LON_MIN = -72.0
LON_MAX = -68.0

# Reducimos el rango de fechas para probar primero que funcione
START_DATE = "2023-01-01"
END_DATE = "2023-12-31" 

def crear_estructura():
    rutas = ["data", "data/modis", "data/modis/csv"]
    for ruta in rutas:
        os.makedirs(ruta, exist_ok=True)
        print(f"[OK] Carpeta creada/verificada: {ruta}")

def inicializar_gee():
    import ee
    try:
        # Intentamos inicializar con tu proyecto de Cloud
        ee.Initialize(project=EE_PROJECT_ID)
        print(f"[OK] Earth Engine inicializado con proyecto: {EE_PROJECT_ID}")
    except Exception as e:
        print("\n[INFO] Necesitas autenticarte por primera vez o el token expiró...")
        # Forzamos la autenticación usando tu proyecto
        ee.Authenticate(project=EE_PROJECT_ID)
        ee.Initialize(project=EE_PROJECT_ID)
        print("[OK] Autenticación completada.")
    return ee

def extraer_modis_a_csv(ee):
    print("\n" + "="*70)
    print("EXTRAYENDO DATOS MODIS DIRECTO A CSV (SIN GEOTIFFS)")
    print("="*70)

    # Definir la geometría de Puno
    region = ee.Geometry.Rectangle([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX])

    # 1. Colección de Temperatura (LST_Night_1km es la crítica para heladas)
    lst_col = (ee.ImageCollection("MODIS/061/MOD11A1")
               .filterDate(START_DATE, END_DATE)
               .filterBounds(region)
               .select(['LST_Night_1km', 'LST_Day_1km']))

    # 2. Colección de Índices de Vegetación (NDVI)
    ndvi_col = (ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate(START_DATE, END_DATE)
                .filterBounds(region)
                .select(['NDVI', 'EVI']))

    print("[INFO] Descargando Temperatura (LST)... Esto puede tardar unos minutos.")
    
    # Extraemos muestras directas usando geemap o API básica
    # Para evitar saturar memoria, sacaremos una muestra representativa
    import geemap
    
    # Extraer valores como un GeoPandas/Pandas DataFrame directamente
    try:
        # Convertimos una imagen promedio mensual para prueba rápida (puedes cambiar a daily luego)
        lst_img = lst_col.mean()
        ndvi_img = ndvi_col.mean()
        
        # Combinamos las bandas
        combined = lst_img.addBands(ndvi_img)
        
        # Extraemos puntos aleatorios dentro de Puno (ej. 1000 puntos para entrenar)
        print("[INFO] Generando puntos de muestreo en Puno...")
        points = ee.FeatureCollection.randomPoints(region=region, points=1000)
        
        # Obtenemos los datos de esos puntos
        print("[INFO] Extrayendo valores de las imágenes a los puntos...")
        sampled_data = combined.sampleRegions(
            collection=points,
            scale=1000, # 1km de resolución
            geometries=True
        )
        
        # Convertir a Pandas usando geemap
        df = geemap.ee_to_pandas(sampled_data)
        
        # Limpieza de datos (Aplicar factores de escala de MODIS)
        if 'LST_Night_1km' in df.columns:
            df['LST_Night_1km'] = df['LST_Night_1km'] * 0.02 - 273.15 # Pasar a Celsius
        if 'LST_Day_1km' in df.columns:
            df['LST_Day_1km'] = df['LST_Day_1km'] * 0.02 - 273.15
        if 'NDVI' in df.columns:
            df['NDVI'] = df['NDVI'] * 0.0001
            
        print(f"[OK] Datos extraídos correctamente: {len(df)} registros.")
        
        # Guardar CSV
        ruta_csv = "data/modis/csv/modis_processed_direct.csv"
        df.to_csv(ruta_csv, index=False)
        print(f"\n[OK] ¡CSV GENERADO EXISTOSAMENTE EN: {ruta_csv}!")
        print(df.head())
        
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] Hubo un problema al procesar los datos espaciales: {e}")

def main():
    print("\n" + "="*70)
    print("MODIS PIPELINE - PREDICCIÓN DE HELADAS (VERSIÓN FIX CLOUD)")
    print("="*70)
    
    if EE_PROJECT_ID == "PON_AQUI_TU_ID_DE_PROYECTO":
        print("\n[❌ ERROR DETENIDO] Tienes que poner tu ID de Proyecto de Google Cloud en la variable EE_PROJECT_ID arriba en el código.")
        return

    crear_estructura()
    ee = inicializar_gee()
    extraer_modis_a_csv(ee)

if __name__ == "__main__":
    main()