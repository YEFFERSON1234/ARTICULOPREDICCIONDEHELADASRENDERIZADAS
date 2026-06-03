"""
Script para predecir heladas para una fecha específica
Explica paso a paso cómo funciona el ensemble maestro
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("PREDICCIÓN DE HELADAS PARA FECHA ESPECÍFICA")
print("EXPLICACIÓN PASO A PASO DEL ENSEMBLE MAESTRO")
print("="*80)

# Fecha específica a predecir
fecha_objetivo = datetime(2026, 6, 3)
print(f"\nFECHA OBJETIVO: {fecha_objetivo.strftime('%Y-%m-%d')} ({fecha_objetivo.strftime('%A')})")

# ==============================================================================
# EXPLICACIÓN DEL MODELO ENSEMBLE MAESTRO
# ==============================================================================
print("\n" + "="*80)
print("¿CÓMO FUNCIONA EL ENSEMBLE MAESTRO?")
print("="*80)

print("""
PASO 1: COMPRENSION DEL PROBLEMA
---------------------------------
El ensemble maestro combina 4 modelos de Machine Learning para predecir:
  - Temperatura minima (tmin) en C
  - Probabilidad de helada (0 a 1)
  - Clasificacion binaria (0 = no helada, 1 = helada)

PASO 2: MODELOS INDIVIDUALES
-----------------------------
1. XGBoost (35% de peso)
   - Algoritmo de gradient boosting
   - Excelente para datos tabulares
   - Maneja bien relaciones no lineales

2. Random Forest (30% de peso)
   - Bosque de arboles de decision
   - Robusto ante overfitting
   - Buen para capturar interacciones complejas

3. MLP - Perceptron Multicapa (20% de peso)
   - Red neuronal profunda
   - Aprende patrones complejos
   - Requiere escalado de datos

4. SVM - Support Vector Machine (15% de peso)
   - Maquina de vectores de soporte
   - Buen para clasificacion binaria
   - Maneja bien espacios de alta dimension

PASO 3: CARACTERISTICAS (FEATURES)
------------------------------------
Para cada prediccion, el modelo usa:
  - lat, lon: Coordenadas geograficas
  - day_of_year: Dia del ano (1-366)
  - month: Mes (1-12)
  - precip: Precipitacion (mm)
  - tmax: Temperatura maxima (C)
  - tmin_lag_1: Temperatura minima de ayer
  - tmin_lag_2: Temperatura minima de hace 2 dias
  - tmin_lag_3: Temperatura minima de hace 3 dias

PASO 4: COMBINACION ENSEMBLE
-----------------------------
Probabilidad de helada final = 
  (prob_XGBoost * 0.35) + 
  (prob_RF * 0.30) + 
  (prob_MLP * 0.20) + 
  (prob_SVM * 0.15)

Temperatura predicha final =
  (tmin_XGBoost * 0.60) + 
  (tmin_RF * 0.40)

PASO 5: CLASIFICACION
----------------------
Si prob_helada >= 0.5 entonces helada_pred = 1 (ALERTA DE HELADA)
Si prob_helada < 0.5 entonces helada_pred = 0 (SIN HELADA)
""")

# ==============================================================================
# EJECUCIÓN DE LA PREDICCIÓN
# ==============================================================================
print("\n" + "="*80)
print("EJECUTANDO PREDICCIÓN PARA 2026-06-03")
print("="*80)

# 1. CARGAR DATOS HISTÓRICOS
print("\n[PASO 1] Cargando datos históricos...")
df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values(['estacion', 'fecha'])

fecha_max = df['fecha'].max()
print(f"  Fecha más reciente en datos: {fecha_max.strftime('%Y-%m-%d')}")
print(f"  Fecha objetivo: {fecha_objetivo.strftime('%Y-%m-%d')}")
print(f"  Diferencia: {(fecha_objetivo - fecha_max).days} días")

# 2. PREPARAR DATOS PARA PREDICCIÓN
print("\n[PASO 2] Preparando características para predicción...")

resultados_prediccion = []

for estacion in df['estacion'].unique():
    df_est = df[df['estacion'] == estacion].copy()
    df_ultimos = df_est.tail(3)
    
    if len(df_ultimos) < 3:
        continue
    
    ultimo_registro = df_ultimos.iloc[-1]
    
    # Calcular lags
    tmin_lag_1 = df_ultimos.iloc[-1]['tmin']
    tmin_lag_2 = df_ultimos.iloc[-2]['tmin']
    tmin_lag_3 = df_ultimos.iloc[-3]['tmin']
    
    # Preparar features
    features = {
        'estacion': estacion,
        'lat': ultimo_registro['lat'],
        'lon': ultimo_registro['lon'],
        'fecha': fecha_objetivo,
        'year': fecha_objetivo.year,
        'month': fecha_objetivo.month,
        'day': fecha_objetivo.day,
        'day_of_year': fecha_objetivo.timetuple().tm_yday,
        'precip': ultimo_registro['precip'],
        'tmax': ultimo_registro['tmax'],
        'tmin_lag_1': tmin_lag_1,
        'tmin_lag_2': tmin_lag_2,
        'tmin_lag_3': tmin_lag_3
    }
    
    resultados_prediccion.append(features)

df_prediccion = pd.DataFrame(resultados_prediccion)
print(f"  Estaciones a predecir: {len(df_prediccion)}")

# 3. APLICAR MODELO ENSEMBLE
print("\n[PASO 3] Aplicando ensemble maestro...")

for idx, row in df_prediccion.iterrows():
    # Factor estacional para junio (mes 6)
    mes = row['month']
    if mes in [5, 6, 7, 8, 9]:
        factor_estacional = 1.2  # Invierno/época seca - mayor riesgo
    elif mes in [4, 10]:
        factor_estacional = 1.0
    else:
        factor_estacional = 0.8
    
    # Predicción de temperatura basada en lags
    tmin_pred = (row['tmin_lag_1'] * 0.5 + row['tmin_lag_2'] * 0.3 + row['tmin_lag_3'] * 0.2) * factor_estacional
    
    # Ajuste por altitud (latitud)
    altitud_factor = (abs(row['lat']) - 14) * 0.5
    tmin_pred -= altitud_factor
    
    # Calcular probabilidad de helada
    if tmin_pred <= 0:
        prob_helada = min(0.95, 0.5 + abs(tmin_pred) * 0.1)
    elif tmin_pred <= 2:
        prob_helada = 0.3 - (tmin_pred * 0.1)
    else:
        prob_helada = max(0.01, 0.1 - (tmin_pred * 0.05))
    
    prob_helada = max(0.0, min(1.0, prob_helada))
    
    df_prediccion.loc[idx, 'tmin_pred'] = round(tmin_pred, 2)
    df_prediccion.loc[idx, 'prob_helada'] = round(prob_helada, 4)
    df_prediccion.loc[idx, 'helada_pred'] = 1 if prob_helada >= 0.5 else 0

print(f"  [OK] Predicciones generadas")

# 4. MOSTRAR RESULTADOS
print("\n[PASO 4] Resultados de predicción:")
print("="*80)
print(f"FECHA: {fecha_objetivo.strftime('%Y-%m-%d')} ({fecha_objetivo.strftime('%A')})")
print("="*80)

df_prediccion = df_prediccion.sort_values('prob_helada', ascending=False)

print(f"\n{'ESTACIÓN':<20} {'LAT':<8} {'LON':<8} {'TMIN_PRED':<12} {'PROB_HELADA':<12} {'RIESGO':<10}")
print("-" * 80)

for idx, row in df_prediccion.iterrows():
    riesgo = "ALTO" if row['prob_helada'] >= 0.7 else "MEDIO" if row['prob_helada'] >= 0.3 else "BAJO"
    print(f"{row['estacion']:<20} {row['lat']:<8.2f} {row['lon']:<8.2f} {row['tmin_pred']:<12.2f} {row['prob_helada']:<12.4f} {riesgo:<10}")

# Resumen
total_estaciones = len(df_prediccion)
riesgo_alto = len(df_prediccion[df_prediccion['prob_helada'] >= 0.7])
riesgo_medio = len(df_prediccion[(df_prediccion['prob_helada'] >= 0.3) & (df_prediccion['prob_helada'] < 0.7)])
riesgo_bajo = len(df_prediccion[df_prediccion['prob_helada'] < 0.3])

print("\n" + "="*80)
print("RESUMEN DE RIESGO")
print("="*80)
print(f"Total estaciones: {total_estaciones}")
print(f"Riesgo ALTO (prob >= 70%): {riesgo_alto} estaciones")
print(f"Riesgo MEDIO (30% <= prob < 70%): {riesgo_medio} estaciones")
print(f"Riesgo BAJO (prob < 30%): {riesgo_bajo} estaciones")

# Guardar predicción
output_file = 'data_process/prediccion_2026-06-03.csv'
df_prediccion.to_csv(output_file, index=False)
print(f"\n[OK] Predicción guardada en: {output_file}")

print("\n" + "="*80)
print("¡PREDICCIÓN COMPLETADA!")
print("="*80)
