"""
Script para generar gráficos estadísticos generales de los entrenamientos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Cargar datos
print("Cargando datos...")
maestro_df = pd.read_csv('data_process/CSV_MAESTRO_CONSOLIDADO.csv')
maestro_df['fecha'] = pd.to_datetime(maestro_df['fecha'])

# Columnas de probabilidades de los modelos
modelos = {
    'XGBoost': 'probabilidad_helada',
    'LSTM': 'prob_helada_lstm',
    'MLP': 'prob_helada_mlp',
    'Prophet': 'prob_helada_prophet',
    'SARIMAX': 'prob_helada_sarimax',
    'CNN1D': 'prob_helada_cnn1d',
    'Random Forest': 'prob_helada_rf',
    'SVM': 'prob_helada_svm'
}

# ==================== GRÁFICO 1: DISTRIBUCIÓN DE PROBABILIDADES ====================
print("Generando Gráfico 1: Distribución de Probabilidades...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribución de Probabilidades de Helada - Comparación de Modelos', 
             fontsize=16, fontweight='bold', y=1.00)

# 1.1 Histograma comparativo
ax = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, len(modelos)))
for idx, (nombre, col) in enumerate(modelos.items()):
    if col in maestro_df.columns:
        datos_validos = maestro_df[col].dropna()
        ax.hist(datos_validos, bins=50, alpha=0.5, label=nombre, color=colors[idx])
ax.set_xlabel('Probabilidad de Helada', fontweight='bold')
ax.set_ylabel('Frecuencia', fontweight='bold')
ax.set_title('Histograma de Probabilidades por Modelo')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 1.2 Box plot comparativo
ax = axes[0, 1]
datos_box = []
labels_box = []
for nombre, col in modelos.items():
    if col in maestro_df.columns:
        datos_validos = maestro_df[col].dropna()
        datos_box.append(datos_validos)
        labels_box.append(nombre)
bp = ax.boxplot(datos_box, labels=labels_box, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_ylabel('Probabilidad de Helada', fontweight='bold')
ax.set_title('Distribución de Probabilidades (Box Plot)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# 1.3 Estadísticas descriptivas
ax = axes[1, 0]
ax.axis('off')
stats_text = "ESTADÍSTICAS DESCRIPTIVAS\n\n"
for nombre, col in modelos.items():
    if col in maestro_df.columns:
        datos = maestro_df[col].dropna()
        stats_text += f"{nombre}:\n"
        stats_text += f"  Mean: {datos.mean():.4f} | Std: {datos.std():.4f}\n"
        stats_text += f"  Min: {datos.min():.4f} | Max: {datos.max():.4f}\n"
        stats_text += f"  Median: {datos.median():.4f}\n\n"
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 1.4 Correlación entre modelos
ax = axes[1, 1]
cols_modelos = [col for col in modelos.values() if col in maestro_df.columns]
nombres_cols_modelos = [nombre for nombre, col in modelos.items() if col in maestro_df.columns]
corr_matrix = maestro_df[cols_modelos].corr()
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(nombres_cols_modelos)))
ax.set_yticks(range(len(nombres_cols_modelos)))
ax.set_xticklabels(nombres_cols_modelos, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(nombres_cols_modelos, fontsize=8)
ax.set_title('Matriz de Correlación entre Modelos')
plt.colorbar(im, ax=ax, label='Correlación')

# Añadir valores en la matriz
for i in range(len(nombres_cols_modelos)):
    for j in range(len(nombres_cols_modelos)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=7)

plt.tight_layout()
plt.savefig('graficos_resultados/16_distribucion_probabilidades.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 1 guardado")
plt.close()

# ==================== GRÁFICO 2: MÉTRICAS DE RENDIMIENTO ====================
print("Generando Gráfico 2: Métricas de Rendimiento...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Métricas de Rendimiento - Comparación de Modelos', 
             fontsize=16, fontweight='bold', y=1.00)

# Calcular métricas para cada modelo
metricas = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'AUC-ROC': []}
nombres_modelos_metricas = []

for nombre, col in modelos.items():
    if col in maestro_df.columns:
        # Convertir probabilidades a predicciones binarias
        y_true = maestro_df['helada'].values
        y_pred_prob = maestro_df[col].values
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        # Calcular métricas
        nombres_modelos_metricas.append(nombre)
        metricas['Accuracy'].append(accuracy_score(y_true, y_pred))
        metricas['Precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metricas['Recall'].append(recall_score(y_true, y_pred, zero_division=0))
        metricas['F1-Score'].append(f1_score(y_true, y_pred, zero_division=0))
        try:
            metricas['AUC-ROC'].append(roc_auc_score(y_true, y_pred_prob))
        except ValueError as e:
            print(f"   [WARNING] No se pudo calcular AUC-ROC para {nombre}: {e}")
            metricas['AUC-ROC'].append(0)

# 2.1 Accuracy
ax = axes[0, 0]
bars = ax.bar(nombres_modelos_metricas, metricas['Accuracy'], color=colors)
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Accuracy por Modelo')
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=45)
for i, (bar, val) in enumerate(zip(bars, metricas['Accuracy'])):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 2.2 Precision vs Recall
ax = axes[0, 1]
x = np.arange(len(nombres_modelos_metricas))
width = 0.35
bars1 = ax.bar(x - width/2, metricas['Precision'], width, label='Precision', color='skyblue')
bars2 = ax.bar(x + width/2, metricas['Recall'], width, label='Recall', color='lightcoral')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Precision vs Recall')
ax.set_xticks(x)
ax.set_xticklabels(nombres_modelos_metricas, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# 2.3 F1-Score
ax = axes[1, 0]
bars = ax.bar(nombres_modelos_metricas, metricas['F1-Score'], color=colors)
ax.set_ylabel('F1-Score', fontweight='bold')
ax.set_title('F1-Score por Modelo')
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=45)
for i, (bar, val) in enumerate(zip(bars, metricas['F1-Score'])):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 2.4 AUC-ROC
ax = axes[1, 1]
bars = ax.bar(nombres_modelos_metricas, metricas['AUC-ROC'], color=colors)
ax.set_ylabel('AUC-ROC', fontweight='bold')
ax.set_title('AUC-ROC por Modelo')
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=45)
for i, (bar, val) in enumerate(zip(bars, metricas['AUC-ROC'])):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('graficos_resultados/17_metricas_rendimiento.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 2 guardado")
plt.close()

# ==================== GRÁFICO 3: ANÁLISIS POR ESTACIÓN ====================
print("Generando Gráfico 3: Análisis por Estación...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análisis de Entrenamientos por Estación Geográfica', 
             fontsize=16, fontweight='bold', y=1.00)

estaciones = maestro_df['estacion'].unique()

# 3.1 Frecuencia de heladas por estación
ax = axes[0, 0]
heladas_por_estacion = []
for est in estaciones:
    data_est = maestro_df[maestro_df['estacion'] == est]
    freq = (data_est['helada'].sum() / len(data_est)) * 100
    heladas_por_estacion.append(freq)
bars = ax.bar(estaciones, heladas_por_estacion, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_ylabel('Frecuencia de Heladas (%)', fontweight='bold')
ax.set_title('Frecuencia de Heladas por Estación')
ax.set_ylim([0, 100])
for bar, val in zip(bars, heladas_por_estacion):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 3.2 Distribución de registros
ax = axes[0, 1]
registros_por_estacion = [len(maestro_df[maestro_df['estacion'] == est]) for est in estaciones]
wedges, texts, autotexts = ax.pie(registros_por_estacion, labels=estaciones, autopct='%1.1f%%',
                                    colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], startangle=90)
ax.set_title('Distribución de Registros por Estación')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 3.3 Temperatura promedio por estación
ax = axes[1, 0]
tmin_por_estacion = [maestro_df[maestro_df['estacion'] == est]['tmin'].mean() for est in estaciones]
tmax_por_estacion = [maestro_df[maestro_df['estacion'] == est]['tmax'].mean() for est in estaciones]
x = np.arange(len(estaciones))
width = 0.35
bars1 = ax.bar(x - width/2, tmin_por_estacion, width, label='Tmin Promedio', color='steelblue')
bars2 = ax.bar(x + width/2, tmax_por_estacion, width, label='Tmax Promedio', color='coral')
ax.set_ylabel('Temperatura (°C)', fontweight='bold')
ax.set_title('Temperatura Promedio por Estación')
ax.set_xticks(x)
ax.set_xticklabels(estaciones)
ax.legend()
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# 3.4 Cobertura de datos por estación y mes
ax = axes[1, 1]
maestro_df['mes'] = maestro_df['fecha'].dt.month
cobertura_data = pd.crosstab(maestro_df['mes'], maestro_df['estacion'])
cobertura_data.plot(kind='line', ax=ax, marker='o', linewidth=2, markersize=6)
ax.set_xlabel('Mes', fontweight='bold')
ax.set_ylabel('Número de Registros', fontweight='bold')
ax.set_title('Cobertura de Datos por Mes y Estación')
ax.legend(title='Estación')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig('graficos_resultados/18_analisis_estaciones.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 3 guardado")
plt.close()

# ==================== GRÁFICO 4: ANÁLISIS TEMPORAL ====================
print("Generando Gráfico 4: Análisis Temporal...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análisis Temporal de Entrenamientos', 
             fontsize=16, fontweight='bold', y=1.00)

# 4.1 Frecuencia de heladas por mes
ax = axes[0, 0]
heladas_por_mes = maestro_df.groupby('mes').agg({
    'helada': ['sum', 'count']
})
heladas_por_mes.columns = ['heladas', 'total']
heladas_por_mes['frecuencia'] = (heladas_por_mes['heladas'] / heladas_por_mes['total']) * 100
meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct']
bars = ax.bar(range(1, len(heladas_por_mes)+1), heladas_por_mes['frecuencia'], 
              color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(heladas_por_mes))))
ax.set_xlabel('Mes', fontweight='bold')
ax.set_ylabel('Frecuencia de Heladas (%)', fontweight='bold')
ax.set_title('Variación Estacional de Heladas')
ax.set_xticks(range(1, len(heladas_por_mes)+1))
ax.set_xticklabels(meses_nombres, rotation=45)
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3, axis='y')

# 4.2 Temperatura mínima promedio por mes
ax = axes[0, 1]
tmin_por_mes = maestro_df.groupby('mes')['tmin'].mean()
tmax_por_mes = maestro_df.groupby('mes')['tmax'].mean()
ax.plot(range(1, len(tmin_por_mes)+1), tmin_por_mes, marker='o', linewidth=2, 
        markersize=8, label='Tmin Promedio', color='steelblue')
ax.plot(range(1, len(tmax_por_mes)+1), tmax_por_mes, marker='s', linewidth=2, 
        markersize=8, label='Tmax Promedio', color='coral')
ax.fill_between(range(1, len(tmin_por_mes)+1), tmin_por_mes, tmax_por_mes, alpha=0.2)
ax.set_xlabel('Mes', fontweight='bold')
ax.set_ylabel('Temperatura (°C)', fontweight='bold')
ax.set_title('Variación de Temperatura Promedio por Mes')
ax.set_xticks(range(1, len(tmin_por_mes)+1))
ax.set_xticklabels(meses_nombres, rotation=45)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

# 4.3 Amplitud térmica promedio por mes
ax = axes[1, 0]
amp_por_mes = maestro_df.groupby('mes')['amp_termica'].mean()
bars = ax.bar(range(1, len(amp_por_mes)+1), amp_por_mes, 
              color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(amp_por_mes))))
ax.set_xlabel('Mes', fontweight='bold')
ax.set_ylabel('Amplitud Térmica Promedio (°C)', fontweight='bold')
ax.set_title('Amplitud Térmica Promedio por Mes')
ax.set_xticks(range(1, len(amp_por_mes)+1))
ax.set_xticklabels(meses_nombres, rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# 4.4 Precipitación promedio por mes
ax = axes[1, 1]
precip_por_mes = maestro_df.groupby('mes')['precip'].mean()
bars = ax.bar(range(1, len(precip_por_mes)+1), precip_por_mes, 
              color=plt.cm.Blues(np.linspace(0.4, 0.9, len(precip_por_mes))))
ax.set_xlabel('Mes', fontweight='bold')
ax.set_ylabel('Precipitación Promedio (mm)', fontweight='bold')
ax.set_title('Precipitación Promedio por Mes')
ax.set_xticks(range(1, len(precip_por_mes)+1))
ax.set_xticklabels(meses_nombres, rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('graficos_resultados/19_analisis_temporal.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 4 guardado")
plt.close()

# ==================== GRÁFICO 5: MATRIZ DE CONFUSIÓN AGREGADA ====================
print("Generando Gráfico 5: Matriz de Confusión Agregada...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Matrices de Confusión - Modelos Principales', 
             fontsize=16, fontweight='bold', y=1.00)

from sklearn.metrics import confusion_matrix

modelos_principales = [
    ('XGBoost', 'probabilidad_helada'),
    ('LSTM', 'prob_helada_lstm'),
    ('Random Forest', 'prob_helada_rf'),
    ('Ensemble Promedio', 'prob_ensemble_promedio')
]

for idx, (nombre_mod, col_mod) in enumerate(modelos_principales):
    ax = axes[idx // 2, idx % 2]
    
    if col_mod in maestro_df.columns:
        y_true = maestro_df['helada'].values
        y_pred_prob = maestro_df[col_mod].values
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalizar para percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        im = ax.imshow(cm_percent, cmap='Blues', vmin=0, vmax=100)
        
        # Añadir valores
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)',
                              ha="center", va="center", color="white" if cm_percent[i, j] > 50 else "black",
                              fontsize=10, fontweight='bold')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Sin Helada', 'Con Helada'])
        ax.set_yticklabels(['Sin Helada', 'Con Helada'])
        ax.set_xlabel('Predicción', fontweight='bold')
        ax.set_ylabel('Real', fontweight='bold')
        ax.set_title(f'Matriz de Confusión - {nombre_mod}')
        plt.colorbar(im, ax=ax, label='Porcentaje (%)')

plt.tight_layout()
plt.savefig('graficos_resultados/20_matrices_confusion_agregadas.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 5 guardado")
plt.close()

# ==================== GRÁFICO 6: VARIABILIDAD Y DESVIACIÓN ====================
print("Generando Gráfico 6: Variabilidad y Desviación...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análisis de Variabilidad en Entrenamientos', 
             fontsize=16, fontweight='bold', y=1.00)

# 6.1 Desviación estándar de probabilidades
ax = axes[0, 0]
std_probs = []
labels_std = []
colors_std = []
for idx, (nombre, col) in enumerate(modelos.items()):
    if col in maestro_df.columns:
        std_probs.append(maestro_df[col].std())
        labels_std.append(nombre)
        colors_std.append(colors[idx])
bars = ax.bar(labels_std, std_probs, color=colors_std)
ax.set_ylabel('Desviación Estándar', fontweight='bold')
ax.set_title('Desviación Estándar de Probabilidades por Modelo')
ax.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, std_probs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 6.2 Varianza de temperaturas por mes
ax = axes[0, 1]
tmin_std_mes = maestro_df.groupby('mes')['tmin'].std()
ax.plot(range(1, len(tmin_std_mes)+1), tmin_std_mes, marker='o', linewidth=2, 
        markersize=8, color='steelblue', label='Desv. Est. Tmin')
ax.fill_between(range(1, len(tmin_std_mes)+1), tmin_std_mes, alpha=0.3)
ax.set_xlabel('Mes', fontweight='bold')
ax.set_ylabel('Desviación Estándar (°C)', fontweight='bold')
ax.set_title('Variabilidad de Temperatura Mínima por Mes')
ax.set_xticks(range(1, len(tmin_std_mes)+1))
ax.set_xticklabels(meses_nombres, rotation=45)
ax.grid(True, alpha=0.3)

# 6.3 Rango intercuartil por estación
ax = axes[1, 0]
iqr_data = []
for est in estaciones:
    data_est = maestro_df[maestro_df['estacion'] == est]['tmin']
    q75, q25 = data_est.quantile(0.75), data_est.quantile(0.25)
    iqr_data.append(q75 - q25)
bars = ax.bar(estaciones, iqr_data, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_ylabel('IQR (°C)', fontweight='bold')
ax.set_title('Rango Intercuartil (IQR) de Tmin por Estación')
for bar, val in zip(bars, iqr_data):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}',
            ha='center', va='bottom', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 6.4 Coeficiente de variación
ax = axes[1, 1]
cv_data = []
cv_labels = []
colors_cv = []
for idx, (nombre, col) in enumerate(modelos.items()):
    if col in maestro_df.columns:
        datos = maestro_df[col].dropna()
        cv = (datos.std() / datos.mean()) * 100 if datos.mean() != 0 else 0
        cv_data.append(cv)
        cv_labels.append(nombre)
        colors_cv.append(colors[idx])
bars = ax.bar(cv_labels, cv_data, color=colors_cv)
ax.set_ylabel('Coeficiente de Variación (%)', fontweight='bold')
ax.set_title('Coeficiente de Variación (CV) por Modelo')
ax.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, cv_data):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('graficos_resultados/21_variabilidad_desviacion.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 6 guardado")
plt.close()

# ==================== RESUMEN ESTADÍSTICO ====================
print("\n" + "="*80)
print("RESUMEN ESTADÍSTICO DE ENTRENAMIENTOS")
print("="*80)

print("\n📊 ESTADÍSTICAS GENERALES DEL DATASET:")
print(f"  • Total de registros: {len(maestro_df):,}")
print(f"  • Período de datos: {maestro_df['fecha'].min().date()} a {maestro_df['fecha'].max().date()}")
print(f"  • Días de entrenamiento: {(maestro_df['fecha'].max() - maestro_df['fecha'].min()).days}")
print(f"  • Número de estaciones: {maestro_df['estacion'].nunique()}")
print(f"  • Número de meses: {maestro_df['mes'].nunique()}")

print("\n❄️  ESTADÍSTICAS DE HELADAS:")
total_heladas = maestro_df['helada'].sum()
print(f"  • Heladas registradas: {int(total_heladas):,}")
print(f"  • Sin heladas: {len(maestro_df) - int(total_heladas):,}")
print(f"  • Frecuencia de heladas: {(total_heladas / len(maestro_df))*100:.2f}%")

print("\n🤖 RENDIMIENTO DE MODELOS:")
for i, nombre in enumerate(nombres_modelos_metricas):
    print(f"\n  {nombre}:")
    print(f"    • Accuracy:  {metricas['Accuracy'][i]:.4f}")
    print(f"    • Precision: {metricas['Precision'][i]:.4f}")
    print(f"    • Recall:    {metricas['Recall'][i]:.4f}")
    print(f"    • F1-Score:  {metricas['F1-Score'][i]:.4f}")
    print(f"    • AUC-ROC:   {metricas['AUC-ROC'][i]:.4f}")

print("\n🌡️  ESTADÍSTICAS METEOROLÓGICAS:")
print(f"  • Tmin promedio: {maestro_df['tmin'].mean():.2f}°C")
print(f"  • Tmax promedio: {maestro_df['tmax'].mean():.2f}°C")
print(f"  • Amplitud promedio: {maestro_df['amp_termica'].mean():.2f}°C")
print(f"  • Precipitación promedio: {maestro_df['precip'].mean():.2f} mm")

print("\n✅ GRÁFICOS GENERADOS:")
print("  ✓ 16_distribucion_probabilidades.png")
print("  ✓ 17_metricas_rendimiento.png")
print("  ✓ 18_analisis_estaciones.png")
print("  ✓ 19_analisis_temporal.png")
print("  ✓ 20_matrices_confusion_agregadas.png")
print("  ✓ 21_variabilidad_desviacion.png")

print("\n" + "="*80)
print("✅ PROCESO COMPLETADO")
print("="*80)
