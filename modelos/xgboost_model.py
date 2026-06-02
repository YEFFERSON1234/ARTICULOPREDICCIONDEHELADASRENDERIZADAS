import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, f1_score

# 1. CARGA
print("Cargando base de datos real de Puno...")
df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
df['fecha'] = pd.to_datetime(df['fecha'])
df['frost'] = (df['tmin'] <= 0).astype(int)
df = df.sort_values(['estacion', 'fecha']).reset_index(drop=True)

# 2. CARACTERÍSTICAS (TEMPORALES Y LAGS)
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

# 3. DIVISIÓN DINÁMICA (Para que no falle si tu data es hasta 2012)
# Usaremos el último año disponible como test
ultimo_anio = df['year'].max()
print(f"Dividiendo datos: Entrenamiento (< {ultimo_anio}) y Prueba (>= {ultimo_anio})")

train_mask = df['year'] < ultimo_anio
test_mask = df['year'] >= ultimo_anio

X_train, X_test = X[train_mask], X[test_mask]
y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]
y_clf_train, y_clf_test = y_clf[train_mask], y_clf[test_mask]

# Verificación para evitar el error de nuevo
if len(X_test) == 0:
    print("¡ERROR! El conjunto de prueba sigue vacío. Revisa los años de tu CSV.")
else:
    # 4. ENTRENAMIENTO
    print("Entrenando modelos...")
    model_reg = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.04, tree_method='hist')
    model_reg.fit(X_train, y_reg_train)
    
    model_clf = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.04, tree_method='hist')
    model_clf.fit(X_train, y_clf_train)

    # 5. RESULTADOS
    y_reg_pred = model_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    print(f"\nÉXITO. RMSE final: {rmse:.2f}°C")

    # GUARDAR PREDICCIONES
    resultados = df[test_mask].copy()
    resultados['tmin_pred'] = y_reg_pred
    resultados['probabilidad_helada'] = model_clf.predict_proba(X_test)[:, 1]
    resultados.to_csv('data_process/predictions.csv', index=False)
    print("Archivo predictions.csv generado para OpenGL.")