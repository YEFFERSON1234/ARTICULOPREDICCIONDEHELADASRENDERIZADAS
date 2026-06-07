import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from utils.data_loading import (
    load_senamhi_data, add_temporal_features, add_lag_features,
    temporal_train_test_split, DEFAULT_FEATURE_COLS,
)
from utils.evaluation import (
    compute_regression_metrics, compute_classification_metrics,
    print_full_results, save_predictions,
)

# 1. CARGA DE DATOS
print("Cargando datos reales para Random Forest...")
df = load_senamhi_data()

# 2. INGENIERIA DE CARACTERISTICAS
df = add_temporal_features(df)
df = add_lag_features(df)
df = df.dropna()

X = df[DEFAULT_FEATURE_COLS]
y_reg = df['tmin']
y_clf = df['frost']

# 3. DIVISION TEMPORAL DINAMICA
train_mask, test_mask = temporal_train_test_split(df)

X_train, X_test = X[train_mask], X[test_mask]
y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]
y_clf_train, y_clf_test = y_clf[train_mask], y_clf[test_mask]

# 4. ENTRENAMIENTO - RANDOM FOREST REGRESSOR
print(f"Entrenando Random Forest Regressor en {len(X_train)} muestras...")
rf_reg = RandomForestRegressor(
    n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
)
rf_reg.fit(X_train, y_reg_train)

# 5. ENTRENAMIENTO - RANDOM FOREST CLASSIFIER
print("Entrenando Random Forest Classifier...")
rf_clf = RandomForestClassifier(
    n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
)
rf_clf.fit(X_train, y_clf_train)

# 6. EVALUACION
y_reg_pred = rf_reg.predict(X_test)
y_clf_pred = rf_clf.predict(X_test)
y_clf_prob = rf_clf.predict_proba(X_test)[:, 1]

reg_metrics = compute_regression_metrics(y_reg_test, y_reg_pred)
clf_metrics = compute_classification_metrics(y_clf_test, y_clf_pred, y_clf_prob)

print_full_results("RANDOM FOREST", reg_metrics, clf_metrics)

# 7. GUARDAR PREDICCIONES PARA EL ENSAMBLE
save_predictions(
    df[test_mask],
    'data_process/predictions_rf.csv',
    extra_cols={
        'tmin_pred_rf': y_reg_pred,
        'prob_frost_rf': y_clf_prob,
    },
    normalize_keys=False,
)
print("Archivo 'predictions_rf.csv' generado para la comparativa final.")
