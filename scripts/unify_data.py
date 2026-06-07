import pandas as pd
import numpy as np
import sys

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("UNIFICACION DE DATOS SENAMHI + ERA5")
print("="*70)

# 1. Cargar datos SENAMHI
print("\n[1/4] Cargando datos SENAMHI...")
senamhi = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
senamhi['fecha'] = pd.to_datetime(senamhi['fecha'])
print(f"   Registros SENAMHI: {len(senamhi)}")

# 2. Cargar datos ERA5 procesados
print("\n[2/4] Cargando datos ERA5 procesados...")
era5 = pd.read_csv('data/era5_procesado_maestro.csv')
era5['fecha'] = pd.to_datetime(era5['fecha'])
print(f"   Registros ERA5: {len(era5)}")

# 3. Redondear coordenadas para hacer el match
print("\n[3/4] Unificando datos por fecha y coordenadas...")
senamhi['lat_grid'] = senamhi['lat'].round(2)
senamhi['lon_grid'] = senamhi['lon'].round(2)
era5['lat_grid'] = era5['latitude'].round(2)
era5['lon_grid'] = era5['longitude'].round(2)

# Hacer el merge manteniendo TODO el SENAMHI (how='left')
dataset_final = pd.merge(senamhi, era5, how='left', on=['fecha', 'lat_grid', 'lon_grid'])

# 4. LIMPIEZA INTELIGENTE: Rellenar huecos de ERA5 con el promedio histórico
print("\n[4/4] Imputando valores faltantes...")
columnas_clima = ['temp_2m_era5', 'precip_era5', 'presion_era5', 'dew_point_era5']
for col in columnas_clima:
    if col in dataset_final.columns:
        dataset_final[col] = dataset_final[col].fillna(dataset_final[col].mean())

# Limpieza de columnas basura
dataset_final = dataset_final.drop(columns=['lat_grid', 'lon_grid', 'number'], errors='ignore')

# Agregar mes para imputación por mes
dataset_final['mes'] = dataset_final['fecha'].dt.month

# Imputación por Mes
for col in columnas_clima:
    if col in dataset_final.columns:
        dataset_final[col] = dataset_final[col].fillna(dataset_final.groupby('mes')[col].transform('mean'))

# Si aún queda algún hueco, rellenar con el promedio global
for col in columnas_clima:
    if col in dataset_final.columns:
        dataset_final[col] = dataset_final[col].fillna(dataset_final[col].mean())

# Eliminar columna mes temporal
dataset_final = dataset_final.drop(columns=['mes'], errors='ignore')

# Verificación
print(f"\n" + "="*70)
print(f"RESULTADOS DE UNIFICACION")
print(f"="*70)
print(f"Total de registros conservados: {dataset_final.shape[0]}")
print(f"Datos vacíos restantes: {dataset_final.isnull().sum().sum()}")
print(f"Columnas finales: {list(dataset_final.columns)}")

# Guardar
dataset_final.to_csv('data_process/dataset_ML_final_completo.csv', index=False)
print(f"\n[OK] Dataset final guardado en: data_process/dataset_ML_final_completo.csv")
print(f"\nPrimeras filas:")
print(dataset_final.head())
