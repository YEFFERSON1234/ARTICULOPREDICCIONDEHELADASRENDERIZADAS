import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import xgboost as xgb
from utils.data_loading import (
    load_senamhi_data, add_temporal_features, add_lag_features,
    temporal_train_test_split, DEFAULT_FEATURE_COLS,
)
from utils.evaluation import compute_regression_metrics, save_predictions

# 1. CARGA
print("Cargando base de datos real de Puno...")
df = load_senamhi_data()

# 2. CARACTERISTICAS (TEMPORALES Y LAGS)
df = add_temporal_features(df)
df = add_lag_features(df)
df = df.dropna()

X = df[DEFAULT_FEATURE_COLS]
y_reg = df['tmin']
y_clf = df['frost']

# 3. DIVISION DINAMICA
ultimo_anio = df['year'].max()
print(f"Dividiendo datos: Entrenamiento (< {ultimo_anio}) y Prueba (>= {ultimo_anio})")

train_mask, test_mask = temporal_train_test_split(df)

X_train, X_test = X[train_mask], X[test_mask]
y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]
y_clf_train, y_clf_test = y_clf[train_mask], y_clf[test_mask]

if len(X_test) == 0:
    print("ERROR! El conjunto de prueba sigue vacio. Revisa los anios de tu CSV.")
else:
    # 4. ENTRENAMIENTO
    print("Entrenando modelos...")
    model_reg = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.04, tree_method='hist')
    model_reg.fit(X_train, y_reg_train)

    model_clf = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.04, tree_method='hist')
    model_clf.fit(X_train, y_clf_train)

    # 5. RESULTADOS
    y_reg_pred = model_reg.predict(X_test)
    metrics = compute_regression_metrics(y_reg_test, y_reg_pred)
    print(f"\nEXITO. RMSE final: {metrics['rmse']:.2f} C")

    # GUARDAR PREDICCIONES
    save_predictions(
        df[test_mask],
        'data_process/predictions.csv',
        extra_cols={
            'tmin_pred': y_reg_pred,
            'probabilidad_helada': model_clf.predict_proba(X_test)[:, 1],
        },
        normalize_keys=False,
    )
    print("Archivo predictions.csv generado para OpenGL.")
