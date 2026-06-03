"""
prophet_model.py
Modelo Prophet de Facebook para predicción de heladas
Corregido: Error de índice solucionado, sin Data Leakage y sincronizado.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_prophet():
    print("="*70)
    print("MODELO PROPHET (CORREGIDO)")
    print("="*70)
    
    try:
        from prophet import Prophet
    except ImportError:
        print("[ERROR] Prophet no está instalado. Instala con: pip install prophet")
        return
    
    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(['estacion', 'fecha']).reset_index(drop=True)
    
    # Usar una sola estación para Prophet
    estacion = df['estacion'].value_counts().index[0]
    print(f"Usando estación: {estacion}")
    df_est = df[df['estacion'] == estacion].copy()
    
    # SOLUCIÓN AL KEYERROR: Conservar 'fecha' explícitamente junto a las numéricas
    numeric_cols = list(df_est.select_dtypes(include=[np.number]).columns)
    if 'fecha' not in numeric_cols:
        cols_a_guardar = ['fecha'] + numeric_cols
        
    df_est = df_est[cols_a_guardar]
    
    # Resample diario limpio (Rellenando vacíos para no romper la frecuencia temporal)
    df_est = df_est.set_index('fecha')
    df_est = df_est.resample('D').mean()
    df_est = df_est.ffill()
    df_est = df_est.reset_index()
    
    # 2. Preparar datos para Prophet & SOLUCIÓN AL DATA LEAKAGE
    print("[2/4] Preparando desfases temporales y estructuras...")
    df_prophet = pd.DataFrame()
    df_prophet['ds'] = df_est['fecha']
    df_prophet['y'] = df_est['tmin']  # Target actual
    
    # Desplazamos los regresores 1 día al pasado (datos de ayer para predecir hoy)
    df_prophet['precip_ayer'] = df_est['precip'].shift(1)
    df_prophet['tmax_ayer'] = df_est['tmax'].shift(1)
    df_prophet['amp_termica_ayer'] = df_est['amp_termica'].shift(1)
    
    # Guardamos las coordenadas para el archivo de salida
    lat_est = df_est['lat'].iloc[0]
    lon_est = df_est['lon'].iloc[0]
    
    # Eliminamos el primer registro debido al NaN del shift
    df_prophet = df_prophet.dropna().reset_index(drop=True)
    
    # Dividir datos de forma sincronizada (Prueba >= 2015)
    df_prophet['year'] = df_prophet['ds'].dt.year
    train_df = df_prophet[df_prophet['year'] < 2015].copy()
    test_df = df_prophet[df_prophet['year'] >= 2015].copy()
    
    print(f"  Entrenamiento ( < 2015): {len(train_df)} días")
    print(f"  Prueba (>= 2015): {len(test_df)} días")
    
    if len(test_df) == 0:
        print("[ERROR] El set de prueba quedó vacío. Revisa las fechas del archivo original.")
        return

    # 3. Entrenar Prophet
    print("[3/4] Entrenando Prophet con regresores históricos...")
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05
        )
        
        # Agregamos los regresores del pasado validados
        model.add_regressor('precip_ayer')
        model.add_regressor('tmax_ayer')
        model.add_regressor('amp_termica_ayer')
        
        model.fit(train_df)
        print("Modelo Prophet entrenado exitosamente.")
    except Exception as e:
        print(f"[ERROR] No se pudo entrenar: {e}")
        return
    
    # 4. Predecir
    print("[4/4] Realizando predicciones sobre matriz de prueba...")
    
    # Para Prophet con regresores, el dataframe "future" debe contener los regresores de los días de prueba
    future = pd.concat([train_df, test_df], ignore_index=True)[['ds', 'precip_ayer', 'tmax_ayer', 'amp_termica_ayer']]
    
    forecast = model.predict(future)
    
    # Extraer exclusivamente el bloque correspondiente a Test
    y_pred = forecast['yhat'].iloc[-len(test_df):].values
    y_test = test_df['y'].values
    
    # Convertir a clasificación binaria de heladas (Umbral meteorológico <= 0°C)
    y_test_helada = (y_test <= 0).astype(int)
    y_pred_helada = (y_pred <= 0).astype(int)
    
    # Métricas
    f1 = f1_score(y_test_helada, y_pred_helada, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n" + "="*70)
    print("RESULTADOS PROPHET")
    print("="*70)
    print(f"F1-Score (heladas): {f1:.4f}")
    print(f"RMSE (temperatura): {rmse:.4f}°C")
    
    # 5. Guardar predicciones normalizadas para tu ensamble
    resultados = pd.DataFrame({
        'fecha': test_df['ds'].dt.strftime('%Y-%m-%d'),
        'tmin_real': y_test,
        'tmin_pred': y_pred,
        'helada_real': y_test_helada,
        'helada_pred': y_pred_helada,
        # Probabilidad logística aproximada basada en la distancia al cero absoluto
        'prob_helada_prophet': (1 / (1 + np.exp(y_pred))).clip(0, 1)
    })
    resultados['lat'] = round(lat_est, 2)
    resultados['lon'] = round(lon_est, 2)
    
    resultados.to_csv('data_process/predictions_prophet.csv', index=False)
    print(f"\n[OK] Predicciones Prophet científicamente válidas guardadas.")

if __name__ == '__main__':
    train_prophet()