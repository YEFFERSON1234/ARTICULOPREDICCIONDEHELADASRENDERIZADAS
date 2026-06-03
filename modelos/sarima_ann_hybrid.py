"""
sarima_ann_hybrid.py
Modelo Híbrido SARIMA+ANN (Artificial Neural Network)
Corregido: Sin Data Leakage, indexación limpia y corrección de tipos NumPy
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
    print("="*70)
    print("MODELO HÍBRIDO SARIMA+ANN (CORREGIDO)")
    print("="*70)
    
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("[ERROR] statsmodels no está instalado. Instala con: pip install statsmodels")
        return
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(['estacion', 'fecha']).reset_index(drop=True)
    
    # Usar una sola estación para el modelo híbrido
    estacion = df['estacion'].value_counts().index[0]
    print(f"Usando estación: {estacion}")
    df_est = df[df['estacion'] == estacion].copy()
    
    # Asegurar selección correcta de columnas antes del remuestreo
    cols_interes = ['fecha', 'tmin', 'precip', 'tmax', 'amp_termica', 'lat', 'lon']
    df_est = df_est[cols_interes].set_index('fecha')
    
    # SOLUCIÓN AL VALUEWARNING: Mantener la continuidad de la serie temporal
    df_est = df_est.resample('D').mean()
    df_est = df_est.ffill()
    df_est.index.freq = 'D'
    
    # 2. Preparar datos & SOLUCIÓN AL DATA LEAKAGE
    print("[2/5] Generando desfases temporales (Lags)...")
    y = df_est['tmin']
    
    # Desplazamos los regresores de la ANN un día al pasado
    df_est['precip_ayer'] = df_est['precip'].shift(1)
    df_est['tmax_ayer'] = df_est['tmax'].shift(1)
    df_est['amp_termica_ayer'] = df_est['amp_termica'].shift(1)
    
    # Eliminamos la primera fila nula provocada por el desplazamiento
    df_est = df_est.dropna()
    y = df_est['tmin']
    exog = df_est[['precip_ayer', 'tmax_ayer', 'amp_termica_ayer']]
    
    # División Temporal Sincronizada (Prueba >= 2015)
    df_est['year'] = df_est.index.year
    train_mask = df_est['year'] < 2015
    test_mask = df_est['year'] >= 2015
    
    y_train, y_test = y[train_mask], y[test_mask]
    exog_train, exog_test = exog[train_mask], exog[test_mask]
    
    print(f"  Entrenamiento (< 2015): {len(y_train)} días")
    print(f"  Prueba (>= 2015): {len(y_test)} días")
    
    if len(y_test) == 0:
        print("[ERROR] El set de prueba está vacío. Verifica los años disponibles.")
        return
        
    # 3. Entrenar Componente Lineal (SARIMA)
    print("[3/5] Entrenando componente lineal SARIMA...")
    try:
        # Modificamos los parámetros para estabilizar la convergencia de la serie
        sarima_model = SARIMAX(
            y_train,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarima_results = sarima_model.fit(disp=False, maxiter=50)
        
        # Forzar herencia del índice cronológico real
        sarima_pred = sarima_results.get_forecast(steps=len(y_test)).predicted_mean
        sarima_pred.index = y_test.index
        print("  SARIMA entrenado con éxito.")
    except Exception as e:
        print(f"[ERROR] Error crítico en SARIMA: {e}")
        return
    
    # 4. Calcular Residuos y Entrenar Componente No Lineal (ANN)
    print("[4/5] Entrenando componente ANN para modelar residuos...")
    
    # Residuos del set de entrenamiento
    sarima_train_pred = sarima_results.predict(start=y_train.index[0], end=y_train.index[-1])
    residuals_train = y_train.values - sarima_train_pred.values
    
    # Escalado de regresores
    scaler = StandardScaler()
    exog_train_scaled = scaler.fit_transform(exog_train)
    exog_test_scaled = scaler.transform(exog_test)
    
    # Red neuronal encargada de predecir las desviaciones del modelo estadístico
    ann_model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        alpha=0.005,
        max_iter=150,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    ann_model.fit(exog_train_scaled, residuals_train)
    print("  ANN entrenada con éxito.")
    
    # 5. Predicción Híbrida Combinada
    print("[5/5] Realizando predicciones híbridas finales...")
    residuals_pred = ann_model.predict(exog_test_scaled)
    
    # Combinación matemática de la estructura híbrida
    y_pred = sarima_pred.values + residuals_pred
    
    # Convertir salidas a etiquetas binarias de heladas (tmin <= 0°C)
    y_test_helada = (y_test.values <= 0).astype(int)
    y_pred_helada = (y_pred <= 0).astype(int)
    
    # Cálculo de métricas sobre vectores limpios de NumPy
    f1 = f1_score(y_test_helada, y_pred_helada, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
    
    print("\n" + "="*70)
    print("RESULTADOS MODELO HÍBRIDO SARIMA+ANN")
    print("="*70)
    print(f"F1-Score (heladas): {f1:.4f}")
    print(f"RMSE (temperatura): {rmse:.4f}°C")
    
    # 6. Exportar Predicciones Alineadas (SOLUCIÓN AL ATTRIBUTERROR)
    # Eliminamos los métodos .values incorrectos mapeando directamente los arreglos
    resultados = pd.DataFrame({
        'fecha': y_test.index.strftime('%Y-%m-%d'),
        'tmin_real': y_test.values,
        'tmin_pred': y_pred,
        'tmin_sarima': sarima_pred.values,
        'residual_ann': residuals_pred,
        'helada_real': y_test_helada,
        'helada_pred': y_pred_helada,
        'prob_helada_hybrid': (1 / (1 + np.exp(y_pred))).clip(0, 1)
    })
    
    resultados['lat'] = round(df_est['lat'].iloc[0], 2)
    resultados['lon'] = round(df_est['lon'].iloc[0], 2)
    
    resultados.to_csv('data_process/predictions_sarima_ann_hybrid.csv', index=False)
    print(f"\n[OK] Predicciones guardadas: data_process/predictions_sarima_ann_hybrid.csv")

if __name__ == '__main__':
    train_sarima_ann_hybrid()