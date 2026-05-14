import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CARGA DE DATOS
print("Cargando datos reales para Random Forest...")
df = pd.read_csv('limpiezadedatos/datos_heladas_puno_REAL.csv')
df['fecha'] = pd.to_datetime(df['fecha'])
df['frost'] = (df['tmin'] <= 0).astype(int)

# 2. INGENIERÍA DE CARACTERÍSTICAS
# Usamos la misma lógica que en XGBoost para que la comparativa sea justa
df['day_of_year'] = df['fecha'].dt.dayofyear
df['month'] = df['fecha'].dt.month
df['year'] = df['fecha'].dt.year

for lag in [1, 2, 3]:
    df[f'tmin_lag_{lag}'] = df.groupby('estacion')['tmin'].shift(lag)

df = df.dropna()

feature_cols = ['lat', 'lon', 'day_of_year', 'month', 'precip', 'tmax', 'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3']
X = df[feature_cols]
y_reg = df['tmin']
y_clf = df['frost']

# 3. DIVISIÓN TEMPORAL DINÁMICA
ultimo_anio = df['year'].max()
train_mask = df['year'] < ultimo_anio
test_mask = df['year'] >= ultimo_anio

X_train, X_test = X[train_mask], X[test_mask]
y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]
y_clf_train, y_clf_test = y_clf[train_mask], y_clf[test_mask]

# 4. ENTRENAMIENTO - RANDOM FOREST REGRESSOR
print(f"Entrenando Random Forest Regressor en {len(X_train)} muestras...")
rf_reg = RandomForestRegressor(
    n_estimators=100, # Menos que XGBoost porque RF es más pesado en memoria
    max_depth=12,
    n_jobs=-1,        # Usa todos tus núcleos de procesador
    random_state=42
)
rf_reg.fit(X_train, y_reg_train)

# 5. ENTRENAMIENTO - RANDOM FOREST CLASSIFIER
print("Entrenando Random Forest Classifier...")
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X_train, y_clf_train)

# 6. EVALUACIÓN
y_reg_pred = rf_reg.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))

y_clf_pred = rf_clf.predict(X_test)
f1_rf = f1_score(y_clf_test, y_clf_pred)
auc_rf = roc_auc_score(y_clf_test, rf_clf.predict_proba(X_test)[:, 1])

print("\n" + "="*40)
print(f"RESULTADOS RANDOM FOREST")
print("="*40)
print(f"RMSE (Temperatura):   {rmse_rf:.2f}°C")
print(f"F1-Score (Heladas):   {f1_rf*100:.2f}%")
print(f"AUC-ROC:              {auc_rf:.4f}")
print("="*40)

# 7. GUARDAR PREDICCIONES PARA EL ENSAMBLE
resultados_rf = df[test_mask].copy()
resultados_rf['tmin_pred_rf'] = y_reg_pred
resultados_rf['prob_frost_rf'] = rf_clf.predict_proba(X_test)[:, 1]

resultados_rf.to_csv('limpiezadedatos/predictions_rf.csv', index=False)
print("Archivo 'predictions_rf.csv' generado para la comparativa final.")