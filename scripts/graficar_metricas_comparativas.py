"""
Script para graficar métricas comparativas de modelos
Genera gráficos exclusivos del archivo metricas_comparativas.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
GRAFICOS_DIR = Path('graficos_resultados')
GRAFICOS_DIR.mkdir(exist_ok=True)

print("="*80)
print("GRAFICANDO MÉTRICAS COMPARATIVAS DE MODELOS")
print("="*80)

# 1. CARGAR DATOS DE MÉTRICAS
print("\n[1/1] Cargando métricas comparativas...")
df = pd.read_csv('data_process/metricas_comparativas.csv')
print(f"  Modelos: {len(df)}")
print(f"  Métricas: {len(df.columns) - 1}")

# 2. LIMPIAR DATOS (reemplazar NaN con 0 para gráficos)
df_clean = df.fillna(0)

# 3. GRÁFICO 1: COMPARACIÓN COMPLETA DE LAS 4 MÉTRICAS
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Comparación de Métricas de Modelos - Ensemble Maestro vs Modelos Individuales', 
             fontsize=16, fontweight='bold')

# RMSE (menor es mejor)
df_rmse = df_clean[['Modelo', 'RMSE (°C)']].copy()
df_rmse = df_rmse[df_rmse['RMSE (°C)'] > 0]  # Solo modelos con RMSE
bars = axes[0, 0].bar(df_rmse['Modelo'], df_rmse['RMSE (°C)'], 
                     color=['steelblue', 'coral', 'forestgreen'], alpha=0.8, edgecolor='black')
axes[0, 0].set_ylabel('RMSE (°C)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('RMSE (menor es mejor)', fontweight='bold', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)
# Añadir etiquetas de valor
for bar in bars:
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# R² (mayor es mejor)
df_r2 = df_clean[['Modelo', 'R²']].copy()
df_r2 = df_r2[df_r2['R²'] > 0]  # Solo modelos con R²
bars = axes[0, 1].bar(df_r2['Modelo'], df_r2['R²'], 
                     color=['steelblue', 'coral', 'forestgreen'], alpha=0.8, edgecolor='black')
axes[0, 1].set_ylabel('R²', fontsize=11, fontweight='bold')
axes[0, 1].set_title('R² (mayor es mejor)', fontweight='bold', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)
# Añadir etiquetas de valor
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# F1-Score (mayor es mejor)
df_f1 = df_clean[['Modelo', 'F1-Score']].copy()
bars = axes[1, 0].bar(df_f1['Modelo'], df_f1['F1-Score'], 
                     color=['steelblue', 'coral', 'forestgreen', 'orange', 'purple'], alpha=0.8, edgecolor='black')
axes[1, 0].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
axes[1, 0].set_title('F1-Score (mayor es mejor)', fontweight='bold', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].set_ylim(0, 1.1)
# Añadir etiquetas de valor
for bar in bars:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# AUC-ROC (mayor es mejor)
df_auc = df_clean[['Modelo', 'AUC-ROC']].copy()
bars = axes[1, 1].bar(df_auc['Modelo'], df_auc['AUC-ROC'], 
                     color=['steelblue', 'coral', 'forestgreen', 'orange', 'purple'], alpha=0.8, edgecolor='black')
axes[1, 1].set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
axes[1, 1].set_title('AUC-ROC (mayor es mejor)', fontweight='bold', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)
axes[1, 1].set_ylim(0, 1.1)
# Añadir etiquetas de valor
for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'metricas_comparativas_completo.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico completo guardado: graficos_resultados/metricas_comparativas_completo.png")
plt.close()

# 4. GRÁFICO 2: RADAR CHART (COMPARACIÓN VISUAL)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Normalizar métricas para radar chart (0-1 escala)
categories = ['RMSE (invertido)', 'R²', 'F1-Score', 'AUC-ROC']
modelos = ['XGBoost', 'Random Forest', 'ENSEMBLE MAESTRO']

# Normalizar RMSE (invertir porque menor es mejor)
max_rmse = df_clean['RMSE (°C)'].max()
df_norm = df_clean.copy()
df_norm['RMSE_norm'] = 1 - (df_clean['RMSE (°C)'] / max_rmse)  # Invertir

# Valores para cada modelo
values_xgb = [
    df_norm[df_norm['Modelo'] == 'XGBoost']['RMSE_norm'].values[0],
    df_norm[df_norm['Modelo'] == 'XGBoost']['R²'].values[0],
    df_norm[df_norm['Modelo'] == 'XGBoost']['F1-Score'].values[0],
    df_norm[df_norm['Modelo'] == 'XGBoost']['AUC-ROC'].values[0]
]

values_rf = [
    df_norm[df_norm['Modelo'] == 'Random Forest']['RMSE_norm'].values[0],
    df_norm[df_norm['Modelo'] == 'Random Forest']['R²'].values[0],
    df_norm[df_norm['Modelo'] == 'Random Forest']['F1-Score'].values[0],
    df_norm[df_norm['Modelo'] == 'Random Forest']['AUC-ROC'].values[0]
]

values_ensemble = [
    df_norm[df_norm['Modelo'] == 'ENSEMBLE MAESTRO']['RMSE_norm'].values[0],
    df_norm[df_norm['Modelo'] == 'ENSEMBLE MAESTRO']['R²'].values[0],
    df_norm[df_norm['Modelo'] == 'ENSEMBLE MAESTRO']['F1-Score'].values[0],
    df_norm[df_norm['Modelo'] == 'ENSEMBLE MAESTRO']['AUC-ROC'].values[0]
]

# Configurar radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
values_xgb += values_xgb[:1]
values_rf += values_rf[:1]
values_ensemble += values_ensemble[:1]
angles += angles[:1]

ax.plot(angles, values_xgb, 'o-', linewidth=2, label='XGBoost', color='steelblue')
ax.fill(angles, values_xgb, alpha=0.25, color='steelblue')
ax.plot(angles, values_rf, 'o-', linewidth=2, label='Random Forest', color='coral')
ax.fill(angles, values_rf, alpha=0.25, color='coral')
ax.plot(angles, values_ensemble, 'o-', linewidth=3, label='ENSEMBLE MAESTRO', color='forestgreen')
ax.fill(angles, values_ensemble, alpha=0.35, color='forestgreen')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.grid(True)
ax.set_title('Radar Chart Comparativo - Modelos con Métricas Completas', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'metricas_radar_chart.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Radar chart guardado: graficos_resultados/metricas_radar_chart.png")
plt.close()

# 5. GRÁFICO 3: TABLA DE COMPARACIÓN VISUAL
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Crear tabla visual
table_data = []
for idx, row in df_clean.iterrows():
    table_data.append([
        row['Modelo'],
        f"{row['RMSE (°C)']:.2f}" if row['RMSE (°C)'] > 0 else 'N/A',
        f"{row['R²']:.3f}" if row['R²'] > 0 else 'N/A',
        f"{row['F1-Score']:.3f}" if row['F1-Score'] > 0 else 'N/A',
        f"{row['AUC-ROC']:.3f}" if row['AUC-ROC'] > 0 else 'N/A'
    ])

table = ax.table(cellText=table_data, 
                 colLabels=['Modelo', 'RMSE (°C)', 'R²', 'F1-Score', 'AUC-ROC'],
                 cellLoc='center',
                 loc='center',
                 colColours=['#4a90e2', '#4a90e2', '#4a90e2', '#4a90e2', '#4a90e2'])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

# Colorear fila del ensemble maestro
for i in range(len(table_data)):
    if table_data[i][0] == 'ENSEMBLE MAESTRO':
        for j in range(5):
            table[(i+1, j)].set_facecolor('#90EE90')  # Verde claro

ax.set_title('Tabla de Métricas Comparativas', fontsize=14, fontweight='bold', pad=20)

plt.savefig(GRAFICOS_DIR / 'metricas_tabla_comparativa.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Tabla comparativa guardada: graficos_resultados/metricas_tabla_comparativa.png")
plt.close()

# 6. GRÁFICO 4: MEJOR MODELO POR MÉTRICA
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Mejor Modelo por Métrica', fontsize=14, fontweight='bold')

# RMSE (menor es mejor)
best_rmse = df_clean[df_clean['RMSE (°C)'] > 0].nsmallest(1, 'RMSE (°C)')
axes[0].bar(best_rmse['Modelo'], best_rmse['RMSE (°C)'], color='gold', alpha=0.8, edgecolor='black')
axes[0].set_title(f'RMSE: {best_rmse["Modelo"].values[0]}', fontweight='bold')
axes[0].set_ylabel('RMSE (°C)')
axes[0].grid(axis='y', alpha=0.3)

# R² (mayor es mejor)
best_r2 = df_clean[df_clean['R²'] > 0].nlargest(1, 'R²')
axes[1].bar(best_r2['Modelo'], best_r2['R²'], color='gold', alpha=0.8, edgecolor='black')
axes[1].set_title(f'R²: {best_r2["Modelo"].values[0]}', fontweight='bold')
axes[1].set_ylabel('R²')
axes[1].grid(axis='y', alpha=0.3)

# F1-Score (mayor es mejor)
best_f1 = df_clean.nlargest(1, 'F1-Score')
axes[2].bar(best_f1['Modelo'], best_f1['F1-Score'], color='gold', alpha=0.8, edgecolor='black')
axes[2].set_title(f'F1-Score: {best_f1["Modelo"].values[0]}', fontweight='bold')
axes[2].set_ylabel('F1-Score')
axes[2].set_ylim(0, 1.1)
axes[2].grid(axis='y', alpha=0.3)

# AUC-ROC (mayor es mejor)
best_auc = df_clean.nlargest(1, 'AUC-ROC')
axes[3].bar(best_auc['Modelo'], best_auc['AUC-ROC'], color='gold', alpha=0.8, edgecolor='black')
axes[3].set_title(f'AUC-ROC: {best_auc["Modelo"].values[0]}', fontweight='bold')
axes[3].set_ylabel('AUC-ROC')
axes[3].set_ylim(0, 1.1)
axes[3].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'metricas_mejor_modelo.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Mejor modelo por métrica guardado: graficos_resultados/metricas_mejor_modelo.png")
plt.close()

print("\n" + "="*80)
print("RESUMEN DE GRÁFICOS GENERADOS")
print("="*80)
print(f"\n[OK] Todos los gráficos guardados en: graficos_resultados/")
print(f"\nGráficos generados:")
print(f"  1. metricas_comparativas_completo.png - 4 subplots con todas las métricas")
print(f"  2. metricas_radar_chart.png - Radar chart comparativo")
print(f"  3. metricas_tabla_comparativa.png - Tabla visual de métricas")
print(f"  4. metricas_mejor_modelo.png - Mejor modelo por cada métrica")
print("\n" + "="*80)
print("¡GRAFICACIÓN COMPLETADA!")
print("="*80)
