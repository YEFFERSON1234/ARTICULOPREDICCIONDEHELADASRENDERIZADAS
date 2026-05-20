import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                             average_precision_score, accuracy_score, f1_score)

# 1. CARGA DE DATOS
df = pd.read_csv('limpiezadedatos/predictions.csv')
df['fecha'] = pd.to_datetime(df['fecha'])

# 2. CÁLCULO DE MÉTRICAS (Los porcentajes que pediste)
y_real = df['frost']
y_pred = (df['tmin_pred'] <= 0).astype(int)

acc = accuracy_score(y_real, y_pred) * 100
f1 = f1_score(y_real, y_pred) * 100
rmse = np.sqrt(((df['tmin'] - df['tmin_pred']) ** 2).mean())

print("="*40)
print(f"MÉTRICAS GLOBALES - DEPARTAMENTO DE PUNO")
print("="*40)
print(f"Precisión General (Accuracy): {acc:.2f}%")
print(f"F1-Score (Balance Alerta):   {f1:.2f}%")
print(f"Error Promedio (RMSE):       {rmse:.2f}°C")
print("="*40)

# 3. CONFIGURACIÓN DEL REPORTE
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
# CORREGIDO: wspace en lugar de wstep
plt.subplots_adjust(hspace=0.3, wspace=0.2) 

# --- A. TENDENCIA GENERAL DEL DEPARTAMENTO ---
df_global = df.groupby('fecha')[['tmin', 'tmin_pred']].mean().reset_index()
axes[0, 0].plot(df_global['fecha'], df_global['tmin'], label='Real (Promedio Puno)', color='royalblue', alpha=0.8)
axes[0, 0].plot(df_global['fecha'], df_global['tmin_pred'], label='Predicho (Promedio Puno)', color='orange', linestyle='--')
axes[0, 0].set_title(f'Desempeño Global en Puno\n(RMSE: {rmse:.2f}°C)', fontsize=15)
axes[0, 0].legend()

# --- B. CURVA ROC (Discriminación de Helada) ---
fpr, tpr, _ = roc_curve(y_real, df['probabilidad_helada'])
roc_auc = auc(fpr, tpr)
axes[0, 1].plot(fpr, tpr, color='darkred', lw=2, label=f'Área ROC = {roc_auc:.2f}')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_title(f'Capacidad de Clasificación\n(ROC AUC: {roc_auc:.2f})', fontsize=15)
axes[0, 1].legend(loc="lower right")

# --- C. CURVA PRECISION-RECALL (Efectividad) ---
precision, recall, _ = precision_recall_curve(y_real, df['probabilidad_helada'])
axes[1, 0].step(recall, precision, color='green', alpha=0.8, where='post')
axes[1, 0].fill_between(recall, precision, alpha=0.2, color='green')
axes[1, 0].set_title(f'Métrica de Alerta Temprana\n(F1-Score: {f1:.2f}%)', fontsize=15)
axes[1, 0].set_xlabel('Recall (Sensibilidad)')
axes[1, 0].set_ylabel('Precision')

# --- D. DISTRIBUCIÓN GEOGRÁFICA DEL ERROR ---
df['abs_error'] = abs(df['tmin'] - df['tmin_pred'])
error_map = axes[1, 1].scatter(df['lon'], df['lat'], c=df['abs_error'], cmap='YlOrRd', s=30, alpha=0.6)
fig.colorbar(error_map, ax=axes[1, 1], label='Error Absoluto (°C)')
axes[1, 1].set_title('Zonas de Incertidumbre en el Mapa', fontsize=15)

plt.savefig('limpiezadedatos/dashboard_puno_final.png', dpi=300)
print("Dashboard guardado en: limpiezadedatos/dashboard_puno_final.png")
plt.show()