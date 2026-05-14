# modelos/regenerar_predicciones_con_coords.py
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from datetime import datetime

print("="*60)
print("REGENERANDO PREDICCIONES CON COORDENADAS")
print("="*60)

# ==========================================
# 1. CARGAR DATOS
# ==========================================
print("\n1. Cargando datos...")
df = pd.read_csv('limpiezadedatos/datos_heladas_altiplano.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['station', 'date']).reset_index(drop=True)

# Features
df['day_of_year'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

for lag in [1, 2, 3]:
    df[f'T2M_MIN_lag_{lag}'] = df.groupby('station')['T2M_MIN'].shift(lag)

df = df.dropna()

feature_cols = [
    'elevation', 'day_of_year', 'month',
    'T2M_MAX', 'T2M_RANGE', 'RH2M', 'WS2M', 'PS', 'PRECTOTCORR',
    'T2M_MIN_lag_1', 'T2M_MIN_lag_2', 'T2M_MIN_lag_3'
]

X = df[feature_cols]
y = df['T2M_MIN']

# ==========================================
# 2. ENTRENAR MODELO
# ==========================================
print("\n2. Entrenando XGBoost...")
train_mask = df['year'] < 2024
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

xgb_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_model.fit(X_train, y_train)
print(f"   RMSE en test: {np.sqrt(np.mean((xgb_model.predict(X_test) - y_test)**2)):.3f}°C")

# ==========================================
# 3. COORDENADAS DE ESTACIONES
# ==========================================
coordenadas = {
    "Puno": (-15.84, -70.02), "Juliaca": (-15.49, -70.13),
    "Azángaro": (-14.90, -70.10), "Ayaviri": (-14.92, -70.59),
    "Macusani": (-14.08, -70.43), "Mazocruz": (-16.75, -69.72),
    "Lampa": (-15.35, -70.37), "Yunguyo": (-16.25, -69.08),
    "Juli": (-16.22, -69.45), "Desaguadero": (-16.57, -69.04),
    "Cojata": (-15.02, -69.37), "Crucero": (-14.35, -70.03),
    "Crucero Alto": (-15.78, -70.92)
}

# ==========================================
# 4. GENERAR PREDICCIONES CON COORDENADAS
# ==========================================
print("\n3. Generando predicciones con coordenadas...")

# Obtener fechas del período de prueba
fechas_test = df[~train_mask]['date'].unique()
fechas_test = np.sort(fechas_test)
print(f"   Período: {pd.to_datetime(fechas_test[0]).date()} a {pd.to_datetime(fechas_test[-1]).date()}")
print(f"   Total días: {len(fechas_test)}")

# Crear carpeta
os.makedirs('limpiezadedatos/predicciones_temporales', exist_ok=True)

# Limpiar archivos anteriores
for f in os.listdir('limpiezadedatos/predicciones_temporales'):
    os.remove(os.path.join('limpiezadedatos/predicciones_temporales', f))

for i, fecha in enumerate(fechas_test):
    fecha_dt = pd.to_datetime(fecha)
    mask_fecha = df['date'] == fecha
    df_fecha = df[mask_fecha].copy()
    
    if len(df_fecha) == 0:
        continue
    
    # Predecir
    X_fecha = df_fecha[feature_cols]
    df_fecha['T2M_MIN_pred'] = xgb_model.predict(X_fecha)
    
    # Calcular probabilidad de helada
    df_fecha['frost_proba'] = 1 / (1 + np.exp(-(0 - df_fecha['T2M_MIN_pred']) / 2))
    
    # Agregar coordenadas
    df_fecha['latitud'] = df_fecha['station'].map(lambda s: coordenadas.get(s, (np.nan, np.nan))[0])
    df_fecha['longitud'] = df_fecha['station'].map(lambda s: coordenadas.get(s, (np.nan, np.nan))[1])
    df_fecha = df_fecha.dropna(subset=['latitud', 'longitud'])
    
    # Guardar con todas las columnas necesarias
    df_out = df_fecha[['station', 'latitud', 'longitud', 'date', 'T2M_MIN_pred', 'frost_proba']]
    
    fecha_str = fecha_dt.strftime('%Y%m%d')
    output_path = f'limpiezadedatos/predicciones_temporales/pred_{fecha_str}.csv'
    df_out.to_csv(output_path, index=False)
    
    if (i + 1) % 50 == 0:
        print(f"   Generados {i+1}/{len(fechas_test)} archivos")

print(f"\n✅ Generados {len(fechas_test)} archivos en limpiezadedatos/predicciones_temporales/")
print("   Ahora ejecuta el visualizador animado")