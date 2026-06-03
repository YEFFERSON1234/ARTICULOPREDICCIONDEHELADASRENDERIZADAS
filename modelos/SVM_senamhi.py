"""
svm_model.py
Modelo SVM (Support Vector Machine) optimizado para datos masivos de estaciones SENAMHI
Corregido: Sin Data Leakage, ejecución veloz mediante aproximación lineal y sincronizado.
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_svm_senamhi():
    print("="*70)
    print("SVM OPTIMIZADO CON DATOS SENAMHI")
    print("="*70)
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos SENAMHI...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Asegurar orden cronológico por estación para aplicar lags de forma correcta
    df = df.sort_values(['estacion', 'fecha']).reset_index(drop=True)
    
    # 2. Ingeniería de características (SOLUCIÓN AL DATA LEAKAGE)
    print("[2/5] Creando características desfasadas (Ayer)...")
    df['day_of_year'] = df['fecha'].dt.dayofyear
    df['month'] = df['fecha'].dt.month
    df['year'] = df['fecha'].dt.year
    
    # Desplazar variables críticas un paso al pasado para romper la correlación directa
    df['tmax_ayer'] = df.groupby('estacion')['tmax'].shift(1)
    df['precip_ayer'] = df.groupby('estacion')['precip'].shift(1)
    df['amp_termica_ayer'] = df.groupby('estacion')['amp_termica'].shift(1)
    
    # Lags históricos de temperatura mínima
    for lag in [1, 2, 3]:
        df[f'tmin_lag_{lag}'] = df.groupby('estacion')['tmin'].shift(lag)
    
    # Remover filas nulas resultantes de los desplazamientos
    df = df.dropna().reset_index(drop=True)
    
    # Variables predictoras sin información del mismo día
    feature_cols = ['lat', 'lon', 'day_of_year', 'month', 
                    'precip_ayer', 'tmax_ayer', 'amp_termica_ayer',
                    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3']
    
    X = df[feature_cols]
    y = df['helada']
    
    # 3. División temporal sincronizada (Prueba >= 2015)
    print("[3/5] Dividiendo datos temporalmente (Corte 2015)...")
    train_mask = df['year'] < 2015
    test_mask = df['year'] >= 2015
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"  Entrenamiento (< 2015): {len(X_train)} muestras")
    print(f"  Prueba (>= 2015): {len(X_test)} muestras")
    
    # 4. Escalado (Mandatorio para algoritmos basados en distancias / hiperplanos)
    print("[4/5] Escalando características...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entrenamiento SVM de Alta Velocidad (Apto para Big Data)
    print("[5/5] Entrenando Clasificador SVM Calibrado...")
    
    # Usamos LinearSVC con penalización por desbalance de clases
    base_svm = LinearSVC(
        C=0.5, 
        class_weight='balanced', 
        random_state=42, 
        dual=False,      # dual=False cuando n_samples > n_features (mejora rendimiento)
        max_iter=2000
    )
    
    # Envolvemos el modelo en un calibrador para poder extraer predict_proba sin congelar la máquina
    svm_model = CalibratedClassifierCV(estimator=base_svm, method='sigmoid', cv=3)
    svm_model.fit(X_train_scaled, y_train)
    print("  Modelo entrenado exitosamente.")
    
    # 6. Evaluación Real
    y_pred = svm_model.predict(X_test_scaled)
    y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n" + "="*70)
    print(f"RESULTADOS SVM")
    print(f"="*70)
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    print(f"\nMatriz de Confusión Real:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nReporte de Clasificación Corregido:")
    print(classification_report(y_test, y_pred))
    
    # 7. Guardar predicciones estructuradas de forma homogénea para el ensamble
    resultados = df[test_mask].copy()
    resultados['prob_helada_svm'] = y_prob
    
    # Normalizar llaves espaciotemporales
    resultados['fecha'] = pd.to_datetime(resultados['fecha']).dt.strftime('%Y-%m-%d')
    resultados['lat'] = resultados['lat'].round(2)
    resultados['lon'] = resultados['lon'].round(2)
    
    resultados.to_csv('data_process/predictions_svm.csv', index=False)
    print(f"\n[OK] Predicciones SVM guardadas en: data_process/predictions_svm.csv")

if __name__ == '__main__':
    train_svm_senamhi()