"""
SVM con datos SENAMHI (no ERA5)
Modelo de Support Vector Machine para predicción de heladas usando datos de estaciones
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_svm_senamhi():
    """Entrena SVM con datos SENAMHI"""
    print("="*70)
    print("SVM CON DATOS SENAMHI")
    print("="*70)
    
    # 1. Cargar datos SENAMHI
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
    
    # 5. Entrenamiento SVM
    print("[5/5] Entrenando SVM...")
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    
    # 6. Evaluación
    y_pred = svm_model.predict(X_test_scaled)
    y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n" + "="*70)
    print(f"RESULTADOS SVM")
    print(f"="*70)
    print(f"F1-Score: {f1:.4f}")
    
    print(f"\nMatriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # 7. Guardar predicciones
    resultados = df[test_mask].copy()
    resultados['prob_helada_svm'] = y_prob
    resultados.to_csv('data_process/predictions_svm.csv', index=False)
    
    print(f"\n[OK] Predicciones SVM guardadas en: data_process/predictions_svm.csv")
    print(f"Total de registros: {len(resultados)}")

if __name__ == '__main__':
    train_svm_senamhi()
