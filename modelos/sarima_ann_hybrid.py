"""
sarima_ann_hybrid.py
Modelo Híbrido SARIMA+ANN (Artificial Neural Network)
Combina modelo estadístico SARIMA con red neuronal para predicción de heladas
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_squared_error
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_sarima_ann_hybrid():
    """Entrena modelo híbrido SARIMA+ANN para predicción de heladas"""
    print("="*70)
    print("MODELO HÍBRIDO SARIMA+ANN")
    print("="*70)
    
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("[ERROR] statsmodels no está instalado")
        print("Instala con: pip install statsmodels")
        return
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')
    
    # Usar una sola estación para el modelo híbrido
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
    print("[2/5] Preparando datos...")
    y = df_est['tmin']
    
    # Variables exógenas para ANN
    exog = df_est[['precip', 'tmax', 'amp_termica']]
    
    # Dividir datos
    train_size = int(len(y) * 0.8)
    y_train = y[:train_size]
    y_test = y[train_size:]
    exog_train = exog[:train_size]
    exog_test = exog[train_size:]
    
    print(f"Entrenamiento: {len(y_train)} días")
    print(f"Prueba: {len(y_test)} días")
    
    # 3. Entrenar SARIMA
    print("[3/5] Entrenando componente SARIMA...")
    try:
        sarima_model = SARIMAX(
            y_train,
            order=(1, 0, 1),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False
        )
        sarima_results = sarima_model.fit(disp=False, maxiter=50)
        sarima_pred = sarima_results.get_forecast(steps=len(y_test)).predicted_mean
        print("SARIMA entrenado")
    except Exception as e:
        print(f"[ERROR] No se pudo entrenar SARIMA: {e}")
        # Usar predicción naive como fallback
        sarima_pred = y_train[-len(y_test):].values
    
    # 4. Calcular residuos y entrenar ANN
    print("[4/5] Entrenando componente ANN para corrección...")
    
    # Calcular residuos de SARIMA en entrenamiento
    sarima_train_pred = sarima_results.predict(start=1, end=len(y_train))
    residuals_train = y_train.values[1:] - sarima_train_pred.values[1:]
    
    # Preparar datos para ANN (predecir residuos)
    # Usar variables exógenas para predecir residuos
    scaler = StandardScaler()
    exog_train_scaled = scaler.fit_transform(exog_train[1:])
    exog_test_scaled = scaler.transform(exog_test)
    
    # Entrenar ANN para predecir residuos
    ann_model = MLPRegressor(
        hidden_layer_sizes=(50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True
    )
    
    ann_model.fit(exog_train_scaled, residuals_train)
    print("ANN entrenada")
    
    # 5. Predecir con modelo híbrido
    print("[5/5] Realizando predicciones híbridas...")
    
    # Predecir residuos con ANN
    residuals_pred = ann_model.predict(exog_test_scaled)
    
    # Combinar predicciones: SARIMA + corrección ANN
    y_pred = sarima_pred.values + residuals_pred
    
    # Calcular helada (tmin <= 0)
    y_test_helada = (y_test <= 0).astype(int)
    y_pred_helada = (y_pred <= 0).astype(int)
    
    # Métricas
    f1 = f1_score(y_test_helada, y_pred_helada)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n" + "="*70)
    print(f"RESULTADOS MODELO HÍBRIDO SARIMA+ANN")
    print(f"="*70)
    print(f"F1-Score (heladas): {f1:.4f}")
    print(f"RMSE (temperatura): {rmse:.4f}°C")
    
    # Guardar predicciones
    resultados = pd.DataFrame({
        'fecha': y_test.index,
        'tmin_real': y_test.values,
        'tmin_pred': y_pred,
        'tmin_sarima': sarima_pred.values,
        'residual_ann': residuals_pred,
        'helada_real': y_test_helada.values,
        'helada_pred': y_pred_helada.values,
        'prob_helada': (1 - y_pred / 10).clip(0, 1)  # Probabilidad aproximada
    })
    resultados['lat'] = df_est['lat'].iloc[0]
    resultados['lon'] = df_est['lon'].iloc[0]
    
    resultados.to_csv('data_process/predictions_sarima_ann_hybrid.csv', index=False)
    print(f"\n[OK] Predicciones híbridas guardadas en: data_process/predictions_sarima_ann_hybrid.csv")

if __name__ == '__main__':
    train_sarima_ann_hybrid()
