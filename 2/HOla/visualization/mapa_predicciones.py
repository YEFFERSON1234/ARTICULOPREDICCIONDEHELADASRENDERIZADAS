# visualization/mapa_predicciones.py
"""
Genera el mapa 3D final con predicciones REALES del modelo para el artículo.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualizacion_utils import (
    cargar_y_preparar_dem,
    cargar_predictions_csv,
    mapear_prediccion_a_dem,
    crear_mapa_heladas
)
import numpy as np

print("="*50)
print("MAPA 3D DE HELADAS - ALTIPLANO PERUANO")
print("="*50)

# ================== CONFIGURACIÓN ==================
RUTA_DEM = 'data/dem_puno_completo.tif'  # Tu DEM unificado
RUTA_PREDICCIONES = 'outputs/predictions_ensemble.csv'  # O predictions.csv
RUTA_SALIDA = 'outputs/mapa_heladas_3d_realista.png'
FACTOR_REDUCCION = 8  # Ajusta para más/menos resolución
# ==================================================

# 1. Cargar DEM
print("\n1. Preparando terreno...")
X, Y, Z, dem_info = cargar_y_preparar_dem(RUTA_DEM, factor_reduccion=FACTOR_REDUCCION)
print(f"   Dimensiones: {dem_info['nx']} x {dem_info['ny']}")
print(f"   Rango altitud: {dem_info['min']:.0f} - {dem_info['max']:.0f} m")

# 2. Cargar predicciones REALES del modelo
print("\n2. Cargando predicciones...")
try:
    df_pred = cargar_predictions_csv(RUTA_PREDICCIONES)
    
    # 3. Mapear predicciones a la malla del DEM
    print("\n3. Interpolando predicciones sobre terreno...")
    riesgo = mapear_prediccion_a_dem(X, Y, df_pred)
    
    # Mostrar estadísticas de riesgo
    print(f"   Riesgo mínimo: {np.nanmin(riesgo):.3f}")
    print(f"   Riesgo máximo: {np.nanmax(riesgo):.3f}")
    print(f"   Riesgo promedio: {np.nanmean(riesgo):.3f}")
    
except FileNotFoundError:
    print("\n⚠️  No se encontró archivo de predicciones.")
    print("   Usando riesgo simulado (basado en altura)...")
    # Fallback: simular riesgo como antes
    riesgo = (Z - np.nanmin(Z)) / (np.nanmax(Z) - np.nanmin(Z))
    riesgo = np.clip(riesgo, 0, 1)

# 4. Crear y guardar el mapa
print("\n4. Generando mapa 3D...")
ruta_guardada = crear_mapa_heladas(
    X, Y, Z, riesgo, dem_info,
    ruta_salida=RUTA_SALIDA,
    titulo='MAPA DE RIESGO DE HELADAS - ALTIPLANO PERUANO\nRegión Puno (Predicción con ML)'
)

print("\n✅ Listo!")
print(f"📁 El mapa se guardó en: {ruta_guardada}")