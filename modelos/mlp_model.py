"""
mlp_model.py
Modelo MLP (Perceptrón Multicapa) para predicción de heladas
Red neuronal profunda con scikit-learn
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
    """Entrena modelo MLP para predicción de heladas"""
    print("="*70)
    print("MODELO MLP (PERCEPTRÓN MULTICAPA)")
    print("="*70)
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos SENAMHI...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 2. Ingeniería de características
    print("[2/5] Creando características...")
    df['day_of_year'] = df['fecha'].dt.dayofyear
    df['month'] = df['fecha'].dt.month
    df['year'] = df['fecha'].dt.year
    
    # Lags temporales
    for lag in [1, 2, 3]:
        df[f'tmin_lag_{lag}'] = df.groupby('estacion')['tmin'].shift(lag)
    
    df = df.dropna()
    
    # Variables predictoras
    feature_cols = ['lat', 'lon', 'day_of_year', 'month', 'precip', 'tmax', 
                    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3', 'amp_termica']
    
    X = df[feature_cols]
    y = df['helada']
    
    # 3. División temporal
    print("[3/5] Dividiendo datos temporalmente...")
    ultimo_anio = df['year'].max()
    train_mask = df['year'] < ultimo_anio
    test_mask = df['year'] >= ultimo_anio
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"  Entrenamiento: {len(X_train)} muestras")
    print(f"  Prueba: {len(X_test)} muestras")
    
    # 4. Escalado
    print("[4/5] Escalando características...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entrenamiento MLP
    print("[5/5] Entrenando MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),  # 3 capas ocultas
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    mlp.fit(X_train_scaled, y_train)
    print("Modelo entrenado exitosamente")
    
    # 6. Evaluación
    y_pred = mlp.predict(X_test_scaled)
    y_prob = mlp.predict_proba(X_test_scaled)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n" + "="*70)
    print(f"RESULTADOS MLP")
    print(f"="*70)
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # 7. Guardar predicciones
    resultados = df[test_mask].copy()
    resultados['prob_helada_mlp'] = y_prob
    resultados.to_csv('data_process/predictions_mlp.csv', index=False)
    
    print(f"\n[OK] Predicciones MLP guardadas en: data_process/predictions_mlp.csv")
    print(f"Total de registros: {len(resultados)}")

if __name__ == '__main__':
    train_mlp()
