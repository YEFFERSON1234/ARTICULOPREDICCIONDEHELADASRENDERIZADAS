import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_squared_error

# 1. CARGA DE RESULTADOS RF
df = pd.read_csv('limpiezadedatos/predictions_rf.csv')
df['fecha'] = pd.to_datetime(df['fecha'])

# Configuración del lienzo
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# --- A. ANÁLISIS DE REGRESIÓN (REAL VS PREDICHO) ---
# Aquí vemos si el promedio de los árboles está cerca de la realidad
axes[0, 0].scatter(df['tmin'], df['tmin_pred_rf'], alpha=0.3, color='purple', s=15)
axes[0, 0].plot([df['tmin'].min(), df['tmin'].max()], [df['tmin'].min(), df['tmin'].max()], 'r--', lw=2)
axes[0, 0].set_title('RF: Correlación Temperatura Real vs Predicha', fontsize=15)
axes[0, 0].set_xlabel('Real (°C)')
axes[0, 0].set_ylabel('Predicho por RF (°C)')

# --- B. DISTRIBUCIÓN DEL ERROR (RESIDUOS) ---
# Nos dice si el modelo tiene algún sesgo constante
error = df['tmin'] - df['tmin_pred_rf']
sns.histplot(error, kde=True, ax=axes[0, 1], color='mediumorchid')
axes[0, 1].set_title(f'Distribución de Errores RF\n(RMSE: {np.sqrt(mean_squared_error(df["tmin"], df["tmin_pred_rf"])):.2f}°C)', fontsize=15)

# --- C. MATRIZ DE CONFUSIÓN (DETECCIÓN DE HELADAS) ---
# Evalúa qué tan bien clasifica el riesgo el "bosque"
y_real = df['frost']
y_pred = (df['tmin_pred_rf'] <= 0).astype(int)
cm = confusion_matrix(y_real, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1, 0])
axes[1, 0].set_title(f'Matriz de Confusión RF\n(F1-Score: {f1_score(y_real, y_pred)*100:.2f}%)', fontsize=15)

# --- D. ZOOM TEMPORAL: CHUQUIBAMBILLA (ZONA NORTE) ---
# Usamos la estación de tu ejemplo para ver cómo modela el día a día
df_zoom = df[df['estacion'] == 'CHUQUIBAMBILLA'].sort_values('fecha').tail(45)
axes[1, 1].plot(df_zoom['fecha'], df_zoom['tmin'], label='Real', marker='o', color='blue')
axes[1, 1].plot(df_zoom['fecha'], df_zoom['tmin_pred_rf'], label='RF Predicho', linestyle='--', color='magenta')
axes[1, 1].axhline(0, color='red', linewidth=1, label='Límite Helada')
axes[1, 1].set_title('Comportamiento en Chuquibambilla (45 días)', fontsize=15)
axes[1, 1].legend()
plt.xticks(rotation=45)

plt.savefig('limpiezadedatos/reporte_random_forest.png', dpi=300)
print("Dashboard guardado exitosamente.")
plt.show()