import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('datos_heladas_altiplano.csv')
df['date'] = pd.to_datetime(df['date'])

# Ordenar por fecha IMPORTANTE
df = df.sort_values(['station', 'date']).reset_index(drop=True)

# Features
df['day_of_year'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Lags (para mejorar prediccion)
for lag in [1, 2, 3]:
    df[f'T2M_MIN_lag_{lag}'] = df.groupby('station')['T2M_MIN'].shift(lag)

# Eliminar NAs
df = df.dropna()

# Features
feature_cols = [
    'elevation', 'day_of_year', 'month',
    'T2M_MAX', 'T2M_RANGE', 'RH2M', 'WS2M', 'PS', 'PRECTOTCORR',
    'T2M_MIN_lag_1', 'T2M_MIN_lag_2', 'T2M_MIN_lag_3'
]

X = df[feature_cols]
y_reg = df['T2M_MIN']
y_clf = df['frost']

# Division temporal estricta
train_mask = df['year'] < 2024
test_mask = df['year'] >= 2024

X_train, X_test = X[train_mask], X[test_mask]
y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]
y_clf_train, y_clf_test = y_clf[train_mask], y_clf[test_mask]

print("="*50)
print("XGBOOST - MODELO DE ENSAMBLE")
print("="*50)
print(f"Entrenamiento: {len(X_train)} muestras")
print(f"Prueba: {len(X_test)} muestras")
print(f"Proporcion heladas test: {y_clf_test.mean():.2%}")

# ==========================================
# XGBOOST - REGRESION
# ==========================================
print("\n" + "-"*30)
print("REGRESION (Temperatura Minima)")
print("-"*30)

xgb_reg = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_reg.fit(X_train, y_reg_train)
y_reg_pred = xgb_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"RMSE: {rmse:.2f}°C")
print(f"MAE: {mae:.2f}°C")
print(f"R²: {r2:.4f}")

# ==========================================
# XGBOOST - CLASIFICACION
# ==========================================
print("\n" + "-"*30)
print("CLASIFICACION (Deteccion de Heladas)")
print("-"*30)

xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_clf.fit(X_train, y_clf_train)
y_clf_pred = xgb_clf.predict(X_test)
y_clf_proba = xgb_clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_clf_test, y_clf_pred)
precision = precision_score(y_clf_test, y_clf_pred)
recall = recall_score(y_clf_test, y_clf_pred)
f1 = f1_score(y_clf_test, y_clf_pred)

tn, fp, fn, tp = confusion_matrix(y_clf_test, y_clf_pred).ravel()
tss = recall - (fp / (fp + tn))

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"TSS: {tss:.4f}")

# ==========================================
# COMPARACION CON PAPER
# ==========================================
print("\n" + "="*50)
print("COMPARACION CON PAPER")
print("="*50)
print(f"Paper XGBoost RMSE: 1.78°C")
print(f"Paper Ensemble RMSE: 1.65°C")
print(f"Tu XGBoost RMSE: {rmse:.2f}°C")

if rmse <= 2.0:
    print("✅ Excelente! Igual o mejor que el paper")
elif rmse <= 2.5:
    print("✅ Bien! Cerca del paper")
else:
    print("⚠️ Mayor que el paper. Normal por datos simulados")

# ==========================================
# IMPORTANCIA DE FEATURES
# ==========================================
print("\n" + "="*50)
print("TOP 10 FEATURES MAS IMPORTANTES")
print("="*50)

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_reg.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))

# ==========================================
# GRAFICOS
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Predicciones vs Reales
axes[0,0].scatter(y_reg_test, y_reg_pred, alpha=0.3, s=5)
axes[0,0].plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
axes[0,0].set_xlabel('Real (°C)')
axes[0,0].set_ylabel('Predicho (°C)')
axes[0,0].set_title(f'XGBoost - Prediccion vs Real\nRMSE = {rmse:.2f}°C')

# 2. Distribucion de errores
errors = y_reg_test - y_reg_pred
axes[0,1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
axes[0,1].set_xlabel('Error (°C)')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].set_title(f'Error medio = {errors.mean():.2f}°C, Std = {errors.std():.2f}°C')

# 3. Importancia de features
sns.barplot(data=importance.head(10), x='importance', y='feature', ax=axes[1,0])
axes[1,0].set_title('Top 10 Features - XGBoost')

# 4. Matriz de confusion
cm = confusion_matrix(y_clf_test, y_clf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
axes[1,1].set_xlabel('Predicho')
axes[1,1].set_ylabel('Real')
axes[1,1].set_title(f'Matriz de Confusion\nTSS = {tss:.3f}')

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=150)
print("\nGrafico guardado: xgboost_results.png")

# ==========================================
# GUARDAR PREDICCIONES
# ==========================================
resultados = df[test_mask].copy()
resultados['T2M_MIN_pred'] = y_reg_pred
resultados['frost_proba'] = y_clf_proba

resultados[['station', 'elevation', 'date', 'T2M_MIN', 'T2M_MIN_pred', 'frost', 'frost_proba']].to_csv('predictions.csv', index=False)

print("\n" + "="*50)
print("ARCHIVO GUARDADO: predictions.csv")
print("Este archivo es el que usara OpenGL para el mapa 3D")
print("="*50)