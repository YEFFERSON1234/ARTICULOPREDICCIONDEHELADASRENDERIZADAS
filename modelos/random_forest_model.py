import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('datos_heladas_altiplano.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Features simples - SOLO las que estarian disponibles en el mundo real
df['day_of_year'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month

feature_cols = ['elevation', 'day_of_year', 'month', 'RH2M', 'WS2M', 'PRECTOTCORR']

X = df[feature_cols]
y = df['T2M_MIN']

# Division temporal
train = df['date'] < pd.Timestamp('2024-01-01')
test = df['date'] >= pd.Timestamp('2024-01-01')

X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Media T2M_MIN train: {y_train.mean():.2f}°C")
print(f"Media T2M_MIN test: {y_test.mean():.2f}°C")

# Modelo baseline: predecir la media
y_baseline = np.full_like(y_test, y_train.mean())
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_baseline))
print(f"\nBaseline (predecir media) RMSE: {rmse_baseline:.2f}°C")

# Random Forest
rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRandom Forest RMSE: {rmse:.2f}°C")
print(f"Random Forest R²: {r2:.4f}")
print(f"Mejora sobre baseline: {(1 - rmse/rmse_baseline)*100:.1f}%")

print(f"\n{'='*50}")
print(f"OBJETIVO del paper: RMSE ≈ 1.83°C")
if 1.5 <= rmse <= 2.5:
    print("✅ Excelente! Tu modelo esta en el rango del paper")
elif rmse < 1.5:
    print("⚠️ RMSE muy bajo. Datos aun demasiado faciles")
else:
    print("⚠️ RMSE muy alto. Revisar")

# Grafico
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.3, s=5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Real (°C)')
plt.ylabel('Predicho (°C)')
plt.title(f'Random Forest - RMSE: {rmse:.2f}°C')
plt.tight_layout()
plt.savefig('rf_results.png', dpi=150)
print("\nGrafico guardado: rf_results.png")