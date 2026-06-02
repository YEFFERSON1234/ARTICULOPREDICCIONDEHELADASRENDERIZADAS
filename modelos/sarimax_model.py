"""
sarimax_model.py
Modelo SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)
Modelo estadístico tradicional para predicción de heladas
"""

import pandas as pd
import numpy as np
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_sarimax():
    """Entrena modelo SARIMAX para predicción de heladas"""
    print("="*70)
    print("MODELO SARIMAX")
    print("="*70)
    
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        print("[ERROR] statsmodels no está instalado")
        print("Instala con: pip install statsmodels")
        return
    
    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')
    
    # Usar una sola estación para SARIMAX (modelo univariado con exógenas)
    estacion = df['estacion'].value_counts().index[0]
    print(f"Usando estación: {estacion}")
    
    df_est = df[df['estacion'] == estacion].copy()
    df_est = df_est.set_index('fecha')
    
    # Seleccionar solo columnas numéricas
    numeric_cols = df_est.select_dtypes(include=[np.number]).columns
    df_est = df_est[numeric_cols]
    
    # Resample a diario
    df_est = df_est.resample('D').mean().dropna()
    
    # 2. Preparar datos
    print("[2/4] Preparando datos...")
    y = df_est['tmin']
    
    # Variables exógenas
    exog = df_est[['precip', 'tmax', 'amp_termica']]
    
    # Dividir datos
    train_size = int(len(y) * 0.8)
    y_train = y[:train_size]
    y_test = y[train_size:]
    exog_train = exog[:train_size]
    exog_test = exog[train_size:]
    
    print(f"Entrenamiento: {len(y_train)} días")
    print(f"Prueba: {len(y_test)} días")
    
    # 3. Entrenar SARIMAX
    print("[3/4] Entrenando SARIMAX (esto puede tardar)...")
    print("Usando parámetros simplificados para velocidad...")
    
    # SARIMAX(p,d,q)(P,D,Q,s)
    # Simplificado: (1,0,1)(1,0,1,365) - estacionalidad anual
    try:
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 365),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = model.fit(disp=False, maxiter=50)
        print("Modelo entrenado exitosamente")
    except Exception as e:
        print(f"[ERROR] No se pudo entrenar SARIMAX: {e}")
        print("Usando versión simplificada sin estacionalidad...")
        try:
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=(1, 0, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False
            )
            results = model.fit(disp=False, maxiter=50)
            print("Modelo simplificado entrenado")
        except Exception as e2:
            print(f"[ERROR] No se pudo entrenar ni versión simplificada: {e2}")
            return
    
    # 4. Predecir
    print("[4/4] Realizando predicciones...")
    predictions = results.get_forecast(steps=len(y_test), exog=exog_test)
    y_pred = predictions.predicted_mean
    
    # Calcular helada (tmin <= 0)
    y_test_helada = (y_test <= 0).astype(int)
    y_pred_helada = (y_pred <= 0).astype(int)
    
    # Métricas
    from sklearn.metrics import f1_score, mean_squared_error
    f1 = f1_score(y_test_helada, y_pred_helada)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n" + "="*70)
    print(f"RESULTADOS SARIMAX")
    print(f"="*70)
    print(f"F1-Score (heladas): {f1:.4f}")
    print(f"RMSE (temperatura): {rmse:.4f}°C")
    
    # Guardar predicciones
    resultados = pd.DataFrame({
        'fecha': y_test.index,
        'tmin_real': y_test.values,
        'tmin_pred': y_pred.values,
        'helada_real': y_test_helada.values,
        'helada_pred': y_pred_helada.values,
        'prob_helada': (1 - y_pred / 10).clip(0, 1)  # Probabilidad aproximada
    })
    resultados['lat'] = df_est['lat'].iloc[0]
    resultados['lon'] = df_est['lon'].iloc[0]
    
    resultados.to_csv('data_process/predictions_sarimax.csv', index=False)
    print(f"\n[OK] Predicciones SARIMAX guardadas en: data_process/predictions_sarimax.csv")

if __name__ == '__main__':
    train_sarimax()
