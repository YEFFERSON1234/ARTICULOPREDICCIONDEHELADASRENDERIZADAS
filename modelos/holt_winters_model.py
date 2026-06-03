"""
holt_winters_model.py
Modelo Holt-Winters (Exponential Smoothing) para predicción de heladas
Modelo estadístico tradicional para series temporales
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_holt_winters():
    """Entrena modelo Holt-Winters para predicción de heladas de forma correcta"""
    print("="*70)
    print("MODELO HOLT-WINTERS (CORREGIDO)")
    print("="*70)
    
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        print("[ERROR] statsmodels no está instalado. Instala con: pip install statsmodels")
        return
    
    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')
    
    # Usar una sola estación 
    estacion = df['estacion'].value_counts().index[0]
    print(f"Usando estación: {estacion}")
    
    df_est = df[df['estacion'] == estacion].copy()
    df_est = df_est.set_index('fecha')
    
    # Seleccionar solo columnas numéricas antes del resample
    numeric_cols = df_est.select_dtypes(include=[np.number]).columns
    df_est = df_est[numeric_cols]
    
    # SOLUCIÓN AL ERROR 1: Resample diario y rellenar nulos en vez de eliminarlos para no perder la frecuencia 'D'
    df_est = df_est.resample('D').mean()
    df_est = df_est.ffill() # Rellena días faltantes con el valor del día anterior
    df_est.index.freq = 'D' # Asigna explícitamente la frecuencia diaria requerida por statsmodels
    
    # 2. Preparar datos
    print("[2/4] Preparando datos...")
    y = df_est['tmin']
    
    # Dividir datos de forma temporal
    train_size = int(len(y) * 0.8)
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    print(f"Entrenamiento: {len(y_train)} días")
    print(f"Prueba: {len(y_test)} días")
    
    # 3. Entrenar Holt-Winters
    print("[3/4] Entrenando Holt-Winters...")
    try:
        # Usamos una inicialización heurística más estable para evitar errores de convergencia
        model = ExponentialSmoothing(
            y_train,
            trend='add',
            seasonal='add',
            seasonal_periods=365,
            damped_trend=True
        )
        results = model.fit(optimized=True, use_brute=True)
        print("Modelo Holt-Winters avanzado entrenado exitosamente")
    except Exception as e:
        print(f"[WARN] No se pudo entrenar con estacionalidad anual: {e}")
        print("Usando versión simplificada...")
        model = ExponentialSmoothing(y_train, trend='add', seasonal=None, damped_trend=True)
        results = model.fit()
        print("Modelo simplificado entrenado")
    
    # 4. Predecir
    print("[4/4] Realizando predicciones...")
    
    # SOLUCIÓN AL ERROR 2: Forzar las predicciones a heredar el índice cronológico real del set de prueba
    y_pred = results.forecast(steps=len(y_test))
    y_pred.index = y_test.index 
    
    # Calcular helada basada en el umbral meteorológico estándar (tmin <= 0°C)
    y_test_helada = (y_test <= 0).astype(int)
    y_pred_helada = (y_pred <= 0).astype(int)
    
    # Métricas finales
    f1 = f1_score(y_test_helada, y_pred_helada, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n" + "="*70)
    print("RESULTADOS HOLT-WINTERS")
    print("="*70)
    print(f"F1-Score (heladas): {f1:.4f}")
    print(f"RMSE (temperatura): {rmse:.4f}°C")
    
    # Guardar predicciones alineadas
    resultados = pd.DataFrame({
        'fecha': y_test.index.strftime('%Y-%m-%d'),
        'tmin_real': y_test.values,
        'tmin_pred': y_pred.values,
        'helada_real': y_test_helada.values,
        'helada_pred': y_pred_helada.values,
        # Escala sigmoide simple para simular probabilidad basada en proximidad a 0°C
        'prob_frost_hw': (1 / (1 + np.exp(y_pred.values))).clip(0, 1)
    })
    
    resultados['lat'] = df_est['lat'].iloc[0]
    resultados['lon'] = df_est['lon'].iloc[0]
    
    resultados.to_csv('data_process/predictions_holt_winters.csv', index=False)
    print(f"\n[OK] Predicciones Holt-Winters guardadas en: data_process/predictions_holt_winters.csv")

if __name__ == '__main__':
    train_holt_winters()
