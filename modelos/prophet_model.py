"""
prophet_model.py
Modelo Prophet de Facebook para predicción de heladas
Modelo de series temporales con componentes estacionales
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
    """Entrena modelo Prophet para predicción de heladas"""
    print("="*70)
    print("MODELO PROPHET")
    print("="*70)
    
    try:
        from prophet import Prophet
    except ImportError:
        print("[ERROR] Prophet no está instalado")
        print("Instala con: pip install prophet")
        return
    
    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')
    
    # Usar una sola estación para Prophet
    estacion = df['estacion'].value_counts().index[0]
    print(f"Usando estación: {estacion}")
    
    df_est = df[df['estacion'] == estacion].copy()
    
    # Seleccionar solo columnas numéricas
    numeric_cols = df_est.select_dtypes(include=[np.number]).columns
    df_est = df_est[numeric_cols]
    
    # Resample a diario
    df_est = df_est.set_index('fecha')
    df_est = df_est.resample('D').mean().dropna()
    df_est = df_est.reset_index()
    
    # 2. Preparar datos para Prophet
    print("[2/4] Preparando datos para Prophet...")
    # Prophet requiere columnas 'ds' (fecha) y 'y' (valor a predecir)
    df_prophet = df_est.rename(columns={'fecha': 'ds', 'tmin': 'y'})
    
    # Agregar regresores adicionales
    df_prophet['precip'] = df_est['precip'].values
    df_prophet['tmax'] = df_est['tmax'].values
    df_prophet['amp_termica'] = df_est['amp_termica'].values
    
    # Dividir datos
    train_size = int(len(df_prophet) * 0.8)
    train_df = df_prophet[:train_size]
    test_df = df_prophet[train_size:]
    
    print(f"Entrenamiento: {len(train_df)} días")
    print(f"Prueba: {len(test_df)} días")
    
    # 3. Entrenar Prophet
    print("[3/4] Entrenando Prophet...")
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            holidays_prior_scale=10
        )
        
        # Agregar regresores
        model.add_regressor('precip')
        model.add_regressor('tmax')
        model.add_regressor('amp_termica')
        
        model.fit(train_df)
        print("Modelo entrenado exitosamente")
    except Exception as e:
        print(f"[ERROR] No se pudo entrenar con regresores: {e}")
        print("Usando versión sin regresores...")
        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive'
            )
            model.fit(train_df)
            print("Modelo simplificado entrenado")
        except Exception as e2:
            print(f"[ERROR] No se pudo entrenar: {e2}")
            return
    
    # 4. Predecir
    print("[4/4] Realizando predicciones...")
    future = model.make_future_dataframe(periods=len(test_df))
    
    # Agregar regresores al futuro si están disponibles
    if 'precip' in test_df.columns:
        future['precip'] = pd.concat([train_df['precip'], test_df['precip']], ignore_index=True)
    if 'tmax' in test_df.columns:
        future['tmax'] = pd.concat([train_df['tmax'], test_df['tmax']], ignore_index=True)
    if 'amp_termica' in test_df.columns:
        future['amp_termica'] = pd.concat([train_df['amp_termica'], test_df['amp_termica']], ignore_index=True)
    
    forecast = model.predict(future)
    
    # Extraer predicciones para el periodo de prueba
    y_pred = forecast['yhat'].iloc[-len(test_df):].values
    y_test = test_df['y'].values
    
    # Calcular helada (tmin <= 0)
    y_test_helada = (y_test <= 0).astype(int)
    y_pred_helada = (y_pred <= 0).astype(int)
    
    # Métricas
    f1 = f1_score(y_test_helada, y_pred_helada)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n" + "="*70)
    print(f"RESULTADOS PROPHET")
    print(f"="*70)
    print(f"F1-Score (heladas): {f1:.4f}")
    print(f"RMSE (temperatura): {rmse:.4f}°C")
    
    # Guardar predicciones
    resultados = pd.DataFrame({
        'fecha': test_df['ds'].values,
        'tmin_real': y_test,
        'tmin_pred': y_pred,
        'helada_real': y_test_helada,
        'helada_pred': y_pred_helada,
        'prob_helada': (1 - y_pred / 10).clip(0, 1)  # Probabilidad aproximada
    })
    resultados['lat'] = df_est['lat'].iloc[train_size]
    resultados['lon'] = df_est['lon'].iloc[train_size]
    
    resultados.to_csv('data_process/predictions_prophet.csv', index=False)
    print(f"\n[OK] Predicciones Prophet guardadas en: data_process/predictions_prophet.csv")

if __name__ == '__main__':
    train_prophet()
