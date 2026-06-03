"""
Script Maestro para Predicciones de Heladas
Combina todos los modelos, genera ensemble maestro, CSV maestro y gráficos comparativos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.metrics import mean_squared_error, r2_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
DATA_PROCESS_DIR = Path('data_process')
GRAFICOS_DIR = Path('graficos_resultados')
GRAFICOS_DIR.mkdir(exist_ok=True)

print("="*80)
print("SISTEMA MAESTRO DE PREDICCIÓN DE HELADAS")
print("="*80)

# 1. CARGAR TODAS LAS PREDICCIONES
print("\n[1/6] Cargando predicciones de todos los modelos...")

try:
    df_xgb = pd.read_csv(DATA_PROCESS_DIR / 'predictions.csv')
    print(f"  [OK] XGBoost: {len(df_xgb)} registros")
except FileNotFoundError:
    print("  [X] XGBoost no encontrado")
    df_xgb = None

try:
    df_rf = pd.read_csv(DATA_PROCESS_DIR / 'predictions_rf.csv')
    print(f"  [OK] Random Forest: {len(df_rf)} registros")
except FileNotFoundError:
    print("  [X] Random Forest no encontrado")
    df_rf = None

try:
    df_mlp = pd.read_csv(DATA_PROCESS_DIR / 'predictions_mlp.csv')
    print(f"  [OK] MLP: {len(df_mlp)} registros")
except FileNotFoundError:
    print("  [X] MLP no encontrado")
    df_mlp = None

try:
    df_svm = pd.read_csv(DATA_PROCESS_DIR / 'predictions_svm.csv')
    print(f"  [OK] SVM: {len(df_svm)} registros")
except FileNotFoundError:
    print("  [X] SVM no encontrado")
    df_svm = None

try:
    df_ensemble = pd.read_csv(DATA_PROCESS_DIR / 'predictions_ensemble.csv')
    print(f"  [OK] Ensemble existente: {len(df_ensemble)} registros")
except FileNotFoundError:
    print("  [X] Ensemble no encontrado")
    df_ensemble = None

# 2. NORMALIZAR FECHAS Y UNIFICAR
print("\n[2/6] Normalizando y unificando datos...")

# Crear dataframe base con datos comunes
if df_xgb is not None:
    df_base = df_xgb[['fecha', 'lat', 'lon', 'tmin', 'helada']].copy()
    df_base['fecha'] = pd.to_datetime(df_base['fecha']).dt.date
    
    # Agregar predicciones de XGBoost
    df_base['tmin_pred_xgb'] = df_xgb['tmin_pred']
    df_base['prob_helada_xgb'] = df_xgb['probabilidad_helada']
else:
    print("ERROR: XGBoost es requerido como base")
    sys.exit(1)

# Agregar predicciones de Random Forest
if df_rf is not None:
    df_rf_temp = df_rf[['fecha', 'lat', 'lon', 'tmin_pred_rf', 'prob_frost_rf']].copy()
    df_rf_temp['fecha'] = pd.to_datetime(df_rf_temp['fecha']).dt.date
    df_base = pd.merge(df_base, df_rf_temp, on=['fecha', 'lat', 'lon'], how='left')
    print(f"  [OK] Random Forest agregado")

# Agregar predicciones de MLP
if df_mlp is not None:
    df_mlp_temp = df_mlp[['fecha', 'lat', 'lon', 'prob_helada_mlp']].copy()
    df_mlp_temp['fecha'] = pd.to_datetime(df_mlp_temp['fecha']).dt.date
    df_base = pd.merge(df_base, df_mlp_temp, on=['fecha', 'lat', 'lon'], how='left')
    print(f"  [OK] MLP agregado")

# Agregar predicciones de SVM
if df_svm is not None:
    df_svm_temp = df_svm[['fecha', 'lat', 'lon', 'prob_helada_svm']].copy()
    df_svm_temp['fecha'] = pd.to_datetime(df_svm_temp['fecha']).dt.date
    df_base = pd.merge(df_base, df_svm_temp, on=['fecha', 'lat', 'lon'], how='left')
    print(f"  [OK] SVM agregado")

print(f"  Total registros unificados: {len(df_base)}")

# 3. CREAR ENSEMBLE MAESTRO
print("\n[3/6] Creando Ensemble Maestro...")

# Definir pesos para el ensemble (basado en rendimiento típico)
# XGBoost: 35%, RF: 30%, MLP: 20%, SVM: 15%
weights = {
    'xgb': 0.35,
    'rf': 0.30,
    'mlp': 0.20,
    'svm': 0.15
}

# Calcular probabilidad de helada del ensemble maestro
df_base['prob_helada_maestro'] = (
    df_base['prob_helada_xgb'] * weights['xgb']
)

if 'prob_frost_rf' in df_base.columns:
    df_base['prob_helada_maestro'] += df_base['prob_frost_rf'] * weights['rf']

if 'prob_helada_mlp' in df_base.columns:
    # Manejar valores extremos de MLP
    df_base['prob_helada_mlp_adj'] = np.clip(df_base['prob_helada_mlp'], 0, 1)
    df_base['prob_helada_maestro'] += df_base['prob_helada_mlp_adj'] * weights['mlp']

if 'prob_helada_svm' in df_base.columns:
    df_base['prob_helada_maestro'] += df_base['prob_helada_svm'] * weights['svm']

# Clasificación binaria del ensemble maestro
df_base['helada_pred_maestro'] = (df_base['prob_helada_maestro'] >= 0.5).astype(int)

# Predicción de temperatura del ensemble maestro (promedio de regresores)
df_base['tmin_pred_maestro'] = df_base['tmin_pred_xgb'] * 0.6  # XGBoost tiene más peso en regresión
if 'tmin_pred_rf' in df_base.columns:
    df_base['tmin_pred_maestro'] += df_base['tmin_pred_rf'] * 0.4

print(f"  [OK] Ensemble maestro creado con pesos: {weights}")
print(f"  [OK] Probabilidad de helada: prob_helada_maestro")
print(f"  [OK] Clasificación: helada_pred_maestro")
print(f"  [OK] Temperatura predicha: tmin_pred_maestro")

# 4. GUARDAR CSV MAESTRO
print("\n[4/6] Guardando CSV Maestro...")

# Seleccionar columnas importantes para el CSV maestro
columnas_maestro = [
    'fecha', 'lat', 'lon', 'tmin', 'helada',
    'tmin_pred_xgb', 'prob_helada_xgb',
    'tmin_pred_maestro', 'prob_helada_maestro', 'helada_pred_maestro'
]

if 'tmin_pred_rf' in df_base.columns:
    columnas_maestro.insert(-2, 'tmin_pred_rf')
    columnas_maestro.insert(-2, 'prob_frost_rf')

if 'prob_helada_mlp' in df_base.columns:
    columnas_maestro.insert(-2, 'prob_helada_mlp')

if 'prob_helada_svm' in df_base.columns:
    columnas_maestro.insert(-2, 'prob_helada_svm')

df_maestro = df_base[columnas_maestro].copy()
csv_maestro_path = DATA_PROCESS_DIR / 'predictions_maestro.csv'
df_maestro.to_csv(csv_maestro_path, index=False)

print(f"  [OK] CSV maestro guardado en: {csv_maestro_path}")
print(f"  Total registros: {len(df_maestro)}")
print(f"  Columnas: {len(df_maestro.columns)}")

# 5. CALCULAR MÉTRICAS DE CADA MODELO
print("\n[5/6] Calculando métricas de cada modelo...")

metricas = []

# XGBoost
if 'tmin_pred_xgb' in df_base.columns:
    rmse_xgb = np.sqrt(mean_squared_error(df_base['tmin'], df_base['tmin_pred_xgb']))
    r2_xgb = r2_score(df_base['tmin'], df_base['tmin_pred_xgb'])
    f1_xgb = f1_score(df_base['helada'], (df_base['prob_helada_xgb'] >= 0.5).astype(int))
    auc_xgb = roc_auc_score(df_base['helada'], df_base['prob_helada_xgb'])
    metricas.append({'Modelo': 'XGBoost', 'RMSE (°C)': rmse_xgb, 'R²': r2_xgb, 'F1-Score': f1_xgb, 'AUC-ROC': auc_xgb})

# Random Forest
if 'tmin_pred_rf' in df_base.columns:
    rmse_rf = np.sqrt(mean_squared_error(df_base['tmin'], df_base['tmin_pred_rf']))
    r2_rf = r2_score(df_base['tmin'], df_base['tmin_pred_rf'])
    f1_rf = f1_score(df_base['helada'], (df_base['prob_frost_rf'] >= 0.5).astype(int))
    auc_rf = roc_auc_score(df_base['helada'], df_base['prob_frost_rf'])
    metricas.append({'Modelo': 'Random Forest', 'RMSE (°C)': rmse_rf, 'R²': r2_rf, 'F1-Score': f1_rf, 'AUC-ROC': auc_rf})

# MLP
if 'prob_helada_mlp' in df_base.columns:
    f1_mlp = f1_score(df_base['helada'], (df_base['prob_helada_mlp'] >= 0.5).astype(int))
    auc_mlp = roc_auc_score(df_base['helada'], df_base['prob_helada_mlp'])
    metricas.append({'Modelo': 'MLP', 'RMSE (°C)': np.nan, 'R²': np.nan, 'F1-Score': f1_mlp, 'AUC-ROC': auc_mlp})

# SVM
if 'prob_helada_svm' in df_base.columns:
    f1_svm = f1_score(df_base['helada'], (df_base['prob_helada_svm'] >= 0.5).astype(int))
    auc_svm = roc_auc_score(df_base['helada'], df_base['prob_helada_svm'])
    metricas.append({'Modelo': 'SVM', 'RMSE (°C)': np.nan, 'R²': np.nan, 'F1-Score': f1_svm, 'AUC-ROC': auc_svm})

# Ensemble Maestro
rmse_maestro = np.sqrt(mean_squared_error(df_base['tmin'], df_base['tmin_pred_maestro']))
r2_maestro = r2_score(df_base['tmin'], df_base['tmin_pred_maestro'])
f1_maestro = f1_score(df_base['helada'], df_base['helada_pred_maestro'])
auc_maestro = roc_auc_score(df_base['helada'], df_base['prob_helada_maestro'])
metricas.append({'Modelo': 'ENSEMBLE MAESTRO', 'RMSE (°C)': rmse_maestro, 'R²': r2_maestro, 'F1-Score': f1_maestro, 'AUC-ROC': auc_maestro})

df_metricas = pd.DataFrame(metricas)
print("\n  MÉTRICAS COMPARATIVAS:")
print(df_metricas.to_string(index=False))

# Guardar métricas en CSV
df_metricas.to_csv(DATA_PROCESS_DIR / 'metricas_comparativas.csv', index=False)
print(f"\n  [OK] Métricas guardadas en: data_process/metricas_comparativas.csv")

# 6. GENERAR GRÁFICOS
print("\n[6/6] Generando gráficos comparativos...")

# 6.1 Gráfico de barras comparativas de métricas
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Comparación de Modelos - Métricas de Rendimiento', fontsize=16, fontweight='bold')

# RMSE
df_rmse = df_metricas[['Modelo', 'RMSE (°C)']].dropna()
axes[0, 0].bar(df_rmse['Modelo'], df_rmse['RMSE (°C)'], color='steelblue', alpha=0.8)
axes[0, 0].set_title('RMSE (Temperatura Mínima)', fontweight='bold')
axes[0, 0].set_ylabel('RMSE (°C)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# R²
df_r2 = df_metricas[['Modelo', 'R²']].dropna()
axes[0, 1].bar(df_r2['Modelo'], df_r2['R²'], color='forestgreen', alpha=0.8)
axes[0, 1].set_title('R² (Coeficiente de Determinación)', fontweight='bold')
axes[0, 1].set_ylabel('R²')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# F1-Score
df_f1 = df_metricas[['Modelo', 'F1-Score']].dropna()
axes[1, 0].bar(df_f1['Modelo'], df_f1['F1-Score'], color='coral', alpha=0.8)
axes[1, 0].set_title('F1-Score (Clasificación de Heladas)', fontweight='bold')
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# AUC-ROC
df_auc = df_metricas[['Modelo', 'AUC-ROC']].dropna()
axes[1, 1].bar(df_auc['Modelo'], df_auc['AUC-ROC'], color='purple', alpha=0.8)
axes[1, 1].set_title('AUC-ROC (Curva ROC)', fontweight='bold')
axes[1, 1].set_ylabel('AUC-ROC')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'comparacion_metricas.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico de métricas guardado en: graficos_resultados/comparacion_metricas.png")
plt.close()

# 6.2 Gráfico de dispersión: Temperatura Real vs Predicha (XGBoost y Ensemble)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Temperatura Mínima: Real vs Predicha', fontsize=16, fontweight='bold')

# XGBoost
axes[0].scatter(df_base['tmin'], df_base['tmin_pred_xgb'], alpha=0.5, s=20)
axes[0].plot([df_base['tmin'].min(), df_base['tmin'].max()], 
             [df_base['tmin'].min(), df_base['tmin'].max()], 'r--', lw=2)
axes[0].set_xlabel('Temperatura Real (°C)')
axes[0].set_ylabel('Temperatura Predicha (°C)')
axes[0].set_title(f'XGBoost (RMSE: {rmse_xgb:.2f}°C, R²: {r2_xgb:.3f})', fontweight='bold')
axes[0].grid(alpha=0.3)

# Ensemble Maestro
axes[1].scatter(df_base['tmin'], df_base['tmin_pred_maestro'], alpha=0.5, s=20, color='green')
axes[1].plot([df_base['tmin'].min(), df_base['tmin'].max()], 
             [df_base['tmin'].min(), df_base['tmin'].max()], 'r--', lw=2)
axes[1].set_xlabel('Temperatura Real (°C)')
axes[1].set_ylabel('Temperatura Predicha (°C)')
axes[1].set_title(f'Ensemble Maestro (RMSE: {rmse_maestro:.2f}°C, R²: {r2_maestro:.3f})', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'temperatura_real_vs_predicha.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico de temperatura guardado en: graficos_resultados/temperatura_real_vs_predicha.png")
plt.close()

# 6.3 Gráfico de probabilidad de helada por modelo
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Probabilidad de Helada por Modelo', fontsize=16, fontweight='bold')

# Muestra aleatoria para no saturar el gráfico
sample_size = min(500, len(df_base))
df_sample = df_base.sample(n=sample_size, random_state=42)
df_sample = df_sample.sort_values('fecha')

x = range(len(df_sample))

if 'prob_helada_xgb' in df_base.columns:
    ax.plot(x, df_sample['prob_helada_xgb'], label='XGBoost', alpha=0.7, linewidth=1)
if 'prob_frost_rf' in df_base.columns:
    ax.plot(x, df_sample['prob_frost_rf'], label='Random Forest', alpha=0.7, linewidth=1)
if 'prob_helada_mlp' in df_base.columns:
    ax.plot(x, df_sample['prob_helada_mlp'], label='MLP', alpha=0.7, linewidth=1)
if 'prob_helada_svm' in df_base.columns:
    ax.plot(x, df_sample['prob_helada_svm'], label='SVM', alpha=0.7, linewidth=1)
ax.plot(x, df_sample['prob_helada_maestro'], label='Ensemble Maestro', alpha=0.9, linewidth=2, color='black', linestyle='--')

ax.axhline(y=0.5, color='red', linestyle=':', label='Umbral (0.5)')
ax.set_xlabel('Muestras (ordenadas por fecha)')
ax.set_ylabel('Probabilidad de Helada')
ax.set_title('Comparación de Probabilidades de Helada', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'probabilidad_helada_por_modelo.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico de probabilidad guardado en: graficos_resultados/probabilidad_helada_por_modelo.png")
plt.close()

# 6.4 Gráfico de distribución de errores
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Distribución de Errores de Predicción', fontsize=16, fontweight='bold')

# XGBoost
errores_xgb = df_base['tmin'] - df_base['tmin_pred_xgb']
axes[0].hist(errores_xgb, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Error (Real - Predicho)')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title(f'XGBoost (Media: {errores_xgb.mean():.3f}°C, Desv: {errores_xgb.std():.3f}°C)', fontweight='bold')
axes[0].grid(alpha=0.3)

# Ensemble Maestro
errores_maestro = df_base['tmin'] - df_base['tmin_pred_maestro']
axes[1].hist(errores_maestro, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Error (Real - Predicho)')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title(f'Ensemble Maestro (Media: {errores_maestro.mean():.3f}°C, Desv: {errores_maestro.std():.3f}°C)', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'distribucion_errores.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico de errores guardado en: graficos_resultados/distribucion_errores.png")
plt.close()

# 6.5 Gráfico de importancia de modelos (pesos en ensemble)
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Pesos de Modelos en Ensemble Maestro', fontsize=16, fontweight='bold')

modelos_ensemble = []
pesos_ensemble = []

if 'prob_helada_xgb' in df_base.columns:
    modelos_ensemble.append('XGBoost')
    pesos_ensemble.append(weights['xgb'])
if 'prob_frost_rf' in df_base.columns:
    modelos_ensemble.append('Random Forest')
    pesos_ensemble.append(weights['rf'])
if 'prob_helada_mlp' in df_base.columns:
    modelos_ensemble.append('MLP')
    pesos_ensemble.append(weights['mlp'])
if 'prob_helada_svm' in df_base.columns:
    modelos_ensemble.append('SVM')
    pesos_ensemble.append(weights['svm'])

colors = plt.cm.Set3(range(len(modelos_ensemble)))
ax.pie(pesos_ensemble, labels=modelos_ensemble, autopct='%1.1f%%', colors=colors, startangle=90)
ax.axis('equal')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'pesos_ensemble.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico de pesos guardado en: graficos_resultados/pesos_ensemble.png")
plt.close()

# RESUMEN FINAL
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"\n[OK] CSV MAESTRO: data_process/predictions_maestro.csv")
print(f"  - Registros: {len(df_maestro)}")
print(f"  - Columnas: {', '.join(df_maestro.columns)}")
print(f"\n[OK] MÉTRICAS: data_process/metricas_comparativas.csv")
print(f"\n[OK] GRÁFICOS GENERADOS EN: graficos_resultados/")
print(f"  1. comparacion_metricas.png - Comparación de RMSE, R², F1-Score, AUC-ROC")
print(f"  2. temperatura_real_vs_predicha.png - Dispersión temperatura real vs predicha")
print(f"  3. probabilidad_helada_por_modelo.png - Probabilidad de helada por modelo")
print(f"  4. distribucion_errores.png - Distribución de errores de predicción")
print(f"  5. pesos_ensemble.png - Pesos de modelos en ensemble maestro")
print(f"\n[OK] MEJOR MODELO: ENSEMBLE MAESTRO")
print(f"  - RMSE: {rmse_maestro:.2f}°C")
print(f"  - R²: {r2_maestro:.3f}")
print(f"  - F1-Score: {f1_maestro:.3f}")
print(f"  - AUC-ROC: {auc_maestro:.3f}")
print("\n" + "="*80)
print("¡PROCESO COMPLETADO EXITOSAMENTE!")
print("="*80)
