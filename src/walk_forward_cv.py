"""
walk_forward_cv.py
Implementación de validación cruzada espaciotemporal walk-forward
Estrategia de validación para datos temporales y espaciales
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def walk_forward_cross_validation(df, n_folds=5, min_train_years=3):
    """
    Implementa validación cruzada walk-forward espaciotemporal
    
    Args:
        df: DataFrame con datos
        n_folds: Número de folds
        min_train_years: Mínimo de años de entrenamiento
    
    Returns:
        dict: Resultados de validación
    """
    print("="*70)
    print("VALIDACIÓN CRUZADA ESPACIOTEMPORAL WALK-FORWARD")
    print("="*70)
    
    # Obtener años únicos
    years = sorted(df['year'].unique())
    print(f"\nAños disponibles: {years}")
    print(f"Total de años: {len(years)}")
    
    # Calcular tamaño de ventana deslizante
    window_size = len(years) // n_folds
    
    results = []
    
    for i in range(n_folds):
        # Definir ventana de entrenamiento y prueba
        train_end_idx = min_train_years + i
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + window_size, len(years))
        
        if test_end_idx <= test_start_idx:
            continue
        
        train_years = years[:train_end_idx]
        test_years = years[test_start_idx:test_end_idx]
        
        print(f"\n--- Fold {i+1}/{n_folds} ---")
        print(f"Entrenamiento: {train_years[0]}-{train_years[-1]} ({len(train_years)} años)")
        print(f"Prueba: {test_years[0]}-{test_years[-1]} ({len(test_years)} años)")
        
        # Dividir datos
        train_mask = df['year'].isin(train_years)
        test_mask = df['year'].isin(test_years)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        # Preparar características
        feature_cols = ['lat', 'lon', 'day_of_year', 'month', 'precip', 'tmax', 
                        'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3', 'amp_termica']
        
        X_train = train_df[feature_cols]
        y_train = train_df['helada']
        X_test = test_df[feature_cols]
        y_test = test_df['helada']
        
        # Entrenar modelo
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Predecir
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        results.append({
            'fold': i + 1,
            'train_years': f"{train_years[0]}-{train_years[-1]}",
            'test_years': f"{test_years[0]}-{test_years[-1]}",
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'f1_score': f1,
            'auc_roc': auc,
            'brier_score': brier
        })
        
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Brier Score: {brier:.4f}")
    
    return results

def main():
    """Función principal"""
    print("\nCargando datos...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['year'] = df['fecha'].dt.year
    df['month'] = df['fecha'].dt.month
    df['day_of_year'] = df['fecha'].dt.dayofyear
    
    # Crear lags
    for lag in [1, 2, 3]:
        df[f'tmin_lag_{lag}'] = df.groupby('estacion')['tmin'].shift(lag)
    
    df = df.dropna()
    
    # Ejecutar validación cruzada
    results = walk_forward_cross_validation(df, n_folds=5, min_train_years=3)
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN DE VALIDACIÓN CRUZADA WALK-FORWARD")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print(f"\nPromedios:")
    print(f"F1-Score: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
    print(f"AUC-ROC: {results_df['auc_roc'].mean():.4f} ± {results_df['auc_roc'].std():.4f}")
    print(f"Brier Score: {results_df['brier_score'].mean():.4f} ± {results_df['brier_score'].std():.4f}")
    
    # Guardar resultados
    results_df.to_csv('data_process/walk_forward_cv_results.csv', index=False)
    print(f"\n[OK] Resultados guardados en: data_process/walk_forward_cv_results.csv")

if __name__ == '__main__':
    main()
