"""
sarimax_model.py
Modelo SARIMAX optimizado para predicción de heladas
Corregido: Sin Data Leakage, ejecución rápida y sincronización temporal
"""

import pandas as pd
import numpy as np
import sys
from sklearn.metrics import f1_score, mean_squared_error

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_sarimax():
    print("="*70)
    print("MODELO SARIMAX (CORREGIDO Y OPTIMIZADO)")
    print("="*70)
    
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("[ERROR] statsmodels no está instalado. Instala con: pip install statsmodels")
        return
    
    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(['estacion', 'fecha']).reset_index(drop=True)
    
    # Usar una sola estación para SARIMAX
    estacion = df['estacion'].value_counts().index[0]
    print(f"Usando estación: {estacion}")
    
    df_est = df[df['estacion'] == estacion].copy()
    
    # Filtrar columnas antes del resampleo para no perder la fecha
    cols_interes = ['fecha', 'tmin', 'precip', 'tmax', 'amp_termica', 'lat', 'lon']
    df_est = df_est[cols_interes].set_index('fecha')
    
    # Asegurar continuidad diaria estricta (Elimina los ValueWarning)
    df_est = df_est.resample('D').mean()
    df_est = df_est.ffill()
    df_est.index.freq = 'D'
    
    # 2. Preparar datos & SOLUCIÓN AL DATA LEAKAGE
    print("[2/4] Generando desfases temporales (Lags)...")
    
    # Pasamos las variables exógenas al pasado (datos de ayer)
    df_est['precip_ayer'] = df_est['precip'].shift(1)
    df_est['tmax_ayer'] = df_est['tmax'].shift(1)
    df_est['amp_termica_ayer'] = df_est['amp_termica'].shift(1)
    
    # Quitar fila nula inicial por el shift
    df_est = df_est.dropna()
    
    y = df_est['tmin']
    exog = df_est[['precip_ayer', 'tmax_ayer', 'amp_termica_ayer']]
    
    # Coordenadas geográficas estables
    lat_est = df_est['lat'].iloc[0]
    lon_est = df_est['lon'].iloc[0]
    
    # División Temporal Sincronizada (Prueba >= 2015)
    df_est['year'] = df_est.index.year
    train_mask = df_est['year'] < 2015
    test_mask = df_est['year'] >= 2015
    
    y_train, y_test = y[train_mask], y[test_mask]
    exog_train, exog_test = exog[train_mask], exog[test_mask]
    
    print(f"  Entrenamiento (< 2015): {len(y_train)} días")
    print(f"  Prueba (>= 2015): {len(y_test)} días")
    
    if len(y_test) == 0:
        print("[ERROR] El set de prueba está vacío. Verifica las fechas.")
        return
    
    # 3. Entrenar SARIMAX (Versión matemática veloz y estable)
    print("[3/4] Entrenando SARIMAX en CPU...")
    try:
        # Nota científica: Al incluir regresores de alta calidad desfasados,
        # no requerimos seasonal_order=365 porque los regresores ya traen la estacionalidad.
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Maxiter limitado a 50 para convergencia rápida
        results = model.fit(disp=False, maxiter=50)
        print("  Modelo entrenado exitosamente.")
    except Exception as e:
        print(f"[ERROR] Error al entrenar SARIMAX: {e}")
        return
    
    # 4. Predecir
    print("[4/4] Realizando predicciones sobre matriz de prueba...")
    predictions = results.get_forecast(steps=len(y_test), exog=exog_test)
    y_pred = predictions.predicted_mean
    y_pred.index = y_test.index  # Sincronizar índices de tiempo
    
    # Clasificación binaria (Umbral <= 0°C)
    y_test_helada = (y_test.values <= 0).astype(int)
    y_pred_helada = (y_pred.values <= 0).astype(int)
    
    # Métricas
    f1 = f1_score(y_test_helada, y_pred_helada, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred.values))
    
    print("\n" + "="*70)
    print("RESULTADOS SARIMAX")
    print("="*70)
    print(f"F1-Score (heladas): {f1:.4f}")
    print(f"RMSE (temperatura): {rmse:.4f}°C")
    
    # 5. Guardar predicciones estructuradas para el ensamble
    resultados = pd.DataFrame({
        'fecha': y_test.index.strftime('%Y-%m-%d'),
        'tmin_real': y_test.values,
        'tmin_pred': y_pred.values,
        'helada_real': y_test_helada,
        'helada_pred': y_pred_helada,
        # Probabilidad suavizada basada en la aproximación de la curva de predicción
        'prob_helada_sarimax': (1 / (1 + np.exp(y_pred.values))).clip(0, 1)
    })
    resultados['lat'] = round(lat_est, 2)
    resultados['lon'] = round(lon_est, 2)
    
    resultados.to_csv('data_process/predictions_sarimax.csv', index=False)
    print(f"\n[OK] Predicciones guardadas en: data_process/predictions_sarimax.csv")

if __name__ == '__main__':
    train_sarimax()