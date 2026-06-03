"""
mlp_model.py
Modelo MLP (Perceptrón Multicapa) para predicción de heladas
Corregido: Sin Data Leakage y sincronizado para el artículo científico
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_mlp():
    print("="*70)
    print("MODELO MLP (PERCEPTRÓN MULTICAPA) - CORREGIDO")
    print("="*70)
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos SENAMHI...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['year'] = df['fecha'].dt.year
    df['month'] = df['fecha'].dt.month
    df['day_of_year'] = df['fecha'].dt.dayofyear
    
    # Asegurar orden cronológico estricto por estación antes de los desplazamientos
    df = df.sort_values(['estacion', 'fecha']).reset_index(drop=True)
    
    # 2. Ingeniería de características (SOLUCIÓN AL DATA LEAKAGE)
    print("[2/5] Creando características desfasadas (Lags)...")
    
    # Desplazamos TODAS las variables climáticas del día actual hacia el pasado (Ayer)
    df['tmax_ayer'] = df.groupby('estacion')['tmax'].shift(1)
    df['precip_ayer'] = df.groupby('estacion')['precip'].shift(1)
    df['amp_termica_ayer'] = df.groupby('estacion')['amp_termica'].shift(1)
    
    # Lags adicionales de temperatura mínima
    for lag in [1, 2, 3]:
        df[f'tmin_lag_{lag}'] = df.groupby('estacion')['tmin'].shift(lag)
    
    # Eliminamos filas con nulos generados por los desplazamientos
    df = df.dropna().reset_index(drop=True)
    
    # Solo variables del pasado + variables geográficas/temporales estables
    feature_cols = ['lat', 'lon', 'day_of_year', 'month', 
                    'precip_ayer', 'tmax_ayer', 'amp_termica_ayer',
                    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3']
    
    X = df[feature_cols]
    y = df['helada']
    
    # 3. División temporal SINCRONIZADA con XGBoost, RF y LSTM (Prueba >= 2015)
    print("[3/5] Dividiendo datos temporalmente (Sincronizado >= 2015)...")
    train_mask = df['year'] < 2015
    test_mask = df['year'] >= 2015
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"  Entrenamiento: {len(X_train)} muestras")
    print(f"  Prueba: {len(X_test)} muestras")
    
    if len(X_test) == 0:
        print("[ERROR] No hay datos en el set de prueba. Revisa los años de tu dataset.")
        return
    
    # 4. Escalado
    print("[4/5] Escalando características...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entrenamiento MLP Optimizado
    print("[5/5] Entrenando MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),  # Arquitectura piramidal eficiente
        activation='relu',
        solver='adam',
        alpha=0.001,                      # Añadimos un poco de regularización L2
        batch_size=256,                   # Definimos tamaño de lote estable para CPU
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=150,
        random_state=42,
        early_stopping=True,              # Evita sobreajuste
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    mlp.fit(X_train_scaled, y_train)
    print("Modelo entrenado exitosamente sin filtración de datos.")
    
    # 6. Evaluación Real
    y_pred = mlp.predict(X_test_scaled)
    y_prob = mlp.predict_proba(X_test_scaled)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n" + "="*70)
    print(f"RESULTADOS REALES MLP")
    print(f"="*70)
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    print(f"\nReporte de Clasificación Corregido:")
    print(classification_report(y_test, y_pred))
    
    # 7. Guardar predicciones con estructura compatible para el ensamble
    resultados = df[test_mask].copy()
    resultados['prob_helada_mlp'] = y_prob
    
    # Formatear llaves de ensamble
    resultados['fecha'] = pd.to_datetime(resultados['fecha']).dt.strftime('%Y-%m-%d')
    resultados['lat'] = resultados['lat'].round(2)
    resultados['lon'] = resultados['lon'].round(2)
    
    resultados.to_csv('data_process/predictions_mlp.csv', index=False)
    print(f"\n[OK] Predicciones reales MLP guardadas en: data_process/predictions_mlp.csv")

if __name__ == '__main__':
    train_mlp()