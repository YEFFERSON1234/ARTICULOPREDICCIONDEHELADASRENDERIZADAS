"""
Script para generar visualizaciones de resultados de predicción de heladas
Incluye: ROC curve, Confusion Matrix, Scatter plots, Comparación de modelos, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, f1_score, accuracy_score
)
import os
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Crear carpeta para guardar gráficos
output_dir = 'graficos_resultados'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Carpeta creada: {output_dir}")

# Ruta base
base_path = 'data_process'

print("="*60)
print("Generando visualizaciones de resultados")
print("="*60)

# 1. CARGAR DATOS PRINCIPALES
print("\n1. Cargando datos...")
predictions_main = pd.read_csv(f'{base_path}/predictions.csv')
y_true = predictions_main['helada'].values
y_pred_proba = predictions_main['probabilidad_helada'].values
y_pred = (y_pred_proba >= 0.5).astype(int)

print(f"   - Datos principales cargados: {len(predictions_main)} registros")

# 2. CARGAR RESULTADOS DE VALIDACIÓN CRUZADA
print("2. Cargando resultados de validación cruzada...")
cv_results = pd.read_csv(f'{base_path}/walk_forward_cv_results.csv')
print(f"   - Folds: {len(cv_results)}")

# 3. CARGAR COMPARACIÓN DE MODELOS
print("3. Cargando comparación de modelos...")
try:
    model_comparison = pd.read_csv(f'{base_path}/comparacion_modelos.csv', index_col=0)
    print(f"   - Modelos: {list(model_comparison.columns)}")
except FileNotFoundError:
    print("   - Archivo de comparación no encontrado")
    model_comparison = None
except Exception as e:
    print(f"   - Error al cargar comparación de modelos: {e}")
    model_comparison = None

# 4. CARGAR PREDICCIONES DE OTROS MODELOS
print("4. Cargando predicciones de otros modelos...")
model_files = {
    'SVM': 'predictions_svm.csv',
    'LSTM': 'predictions_lstm.csv',
    'MLP': 'predictions_mlp.csv',
    'Prophet': 'predictions_prophet.csv',
    'SARIMAX': 'predictions_sarimax.csv',
    'CNN1D': 'predictions_cnn1d.csv',
    'Ensemble': 'predictions_ensemble.csv',
    'RF': 'predictions_rf.csv'
}

model_predictions = {}
for model_name, filename in model_files.items():
    try:
        df = pd.read_csv(f'{base_path}/{filename}')
        if 'prob_helada_svm' in df.columns:
            model_predictions[model_name] = df['prob_helada_svm'].values
        elif 'probabilidad_helada' in df.columns or 'prob_helada' in df.columns:
            prob_col = [col for col in df.columns if 'prob' in col.lower()][0]
            model_predictions[model_name] = df[prob_col].values
        print(f"   - {model_name}: OK")
    except FileNotFoundError:
        print(f"   - {model_name}: Archivo no encontrado ({filename})")
    except Exception as e:
        print(f"   - {model_name}: Error al cargar - {e}")

# ============================================================================
# GRÁFICO 1: CURVA ROC
# ============================================================================
print("\n5. Generando Curva ROC...")
fig, ax = plt.subplots(figsize=(10, 8))

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')

# Adicionar curvas ROC de otros modelos si están disponibles
colors = plt.cm.tab10(np.linspace(0, 1, len(model_predictions)))
for idx, (model_name, y_pred_model) in enumerate(model_predictions.items()):
    try:
        fpr_m, tpr_m, _ = roc_curve(y_true, y_pred_model)
        roc_auc_m = auc(fpr_m, tpr_m)
        ax.plot(fpr_m, tpr_m, linewidth=2, label=f'{model_name} (AUC = {roc_auc_m:.4f})', 
                color=colors[idx], alpha=0.7)
    except ValueError as e:
        print(f"   [WARNING] No se pudo generar curva ROC para {model_name}: {e}")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Curva ROC - Predicción de Heladas', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_curva_roc.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 01_curva_roc.png")
plt.close()

# ============================================================================
# GRÁFICO 2: MATRIZ DE CONFUSIÓN
# ============================================================================
print("\n6. Generando Matriz de Confusión...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz principal
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            cbar_kws={'label': 'Count'}, annot_kws={'size': 12})
axes[0].set_xlabel('Predicción', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Real', fontsize=11, fontweight='bold')
axes[0].set_title('Matriz de Confusión', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(['Sin Helada', 'Con Helada'])
axes[0].set_yticklabels(['Sin Helada', 'Con Helada'])

# Matriz normalizada
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            cbar_kws={'label': 'Porcentaje'}, annot_kws={'size': 12})
axes[1].set_xlabel('Predicción', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Real', fontsize=11, fontweight='bold')
axes[1].set_title('Matriz de Confusión (Normalizada)', fontsize=12, fontweight='bold')
axes[1].set_xticklabels(['Sin Helada', 'Con Helada'])
axes[1].set_yticklabels(['Sin Helada', 'Con Helada'])

plt.tight_layout()
plt.savefig(f'{output_dir}/02_matriz_confusion.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 02_matriz_confusion.png")
plt.close()

# ============================================================================
# GRÁFICO 3: SCATTER PLOT - Predicciones vs Realidad
# ============================================================================
print("\n7. Generando Scatter Plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot con colores por clase real
scatter1 = axes[0].scatter(y_true[y_true == 0], y_pred_proba[y_true == 0], 
                           alpha=0.6, c='blue', label='Sin Helada (Real)', s=30)
scatter2 = axes[0].scatter(y_true[y_true == 1], y_pred_proba[y_true == 1], 
                           alpha=0.6, c='red', label='Con Helada (Real)', s=30)
axes[0].axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
axes[0].set_xlabel('Valor Real', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Probabilidad Predicha', fontsize=11, fontweight='bold')
axes[0].set_title('Predicciones vs Realidad (Todos los datos)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_ylim([-0.05, 1.05])

# Histograma de probabilidades predichas por clase
axes[1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.6, label='Sin Helada (Real)', color='blue')
axes[1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.6, label='Con Helada (Real)', color='red')
axes[1].axvline(x=0.5, color='k', linestyle='--', linewidth=1.5, label='Threshold')
axes[1].set_xlabel('Probabilidad Predicha', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
axes[1].set_title('Distribución de Probabilidades por Clase', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/03_scatter_plot.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 03_scatter_plot.png")
plt.close()

# ============================================================================
# GRÁFICO 4: PRECISION-RECALL CURVE
# ============================================================================
print("\n8. Generando Curva Precision-Recall...")
fig, ax = plt.subplots(figsize=(10, 8))

precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
pr_auc = auc(recall, precision)

ax.plot(recall, precision, 'b-', linewidth=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})')
ax.fill_between(recall, precision, alpha=0.2)

# Adicionar curvas PR de otros modelos
for idx, (model_name, y_pred_model) in enumerate(model_predictions.items()):
    try:
        precision_m, recall_m, _ = precision_recall_curve(y_true, y_pred_model)
        pr_auc_m = auc(recall_m, precision_m)
        ax.plot(recall_m, precision_m, linewidth=2, label=f'{model_name} (AUC = {pr_auc_m:.4f})',
                color=colors[idx], alpha=0.7)
    except ValueError as e:
        print(f"   [WARNING] No se pudo generar curva PR para {model_name}: {e}")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
ax.legend(loc="lower left", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 04_precision_recall_curve.png")
plt.close()

# ============================================================================
# GRÁFICO 5: MÉTRICAS POR FOLD (VALIDACIÓN CRUZADA)
# ============================================================================
print("\n9. Generando gráfico de métricas por Fold...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

folds = np.arange(1, len(cv_results) + 1)

# F1 Score
axes[0, 0].bar(folds, cv_results['f1_score'], color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('F1 Score', fontsize=11, fontweight='bold')
axes[0, 0].set_title('F1 Score por Fold', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim([0.85, 0.95])
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(cv_results['f1_score']):
    axes[0, 0].text(i+1, v+0.002, f'{v:.4f}', ha='center', fontsize=9)

# AUC-ROC
axes[0, 1].bar(folds, cv_results['auc_roc'], color='seagreen', alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
axes[0, 1].set_title('AUC-ROC por Fold', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim([0.95, 0.99])
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(cv_results['auc_roc']):
    axes[0, 1].text(i+1, v+0.002, f'{v:.4f}', ha='center', fontsize=9)

# Brier Score
axes[1, 0].bar(folds, cv_results['brier_score'], color='coral', alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Brier Score', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Fold', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Brier Score por Fold', fontsize=12, fontweight='bold')
axes[1, 0].set_ylim([0.05, 0.08])
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(cv_results['brier_score']):
    axes[1, 0].text(i+1, v+0.002, f'{v:.4f}', ha='center', fontsize=9)

# Número de muestras
axes[1, 1].bar(folds, cv_results['test_samples'], color='mediumpurple', alpha=0.7, 
               edgecolor='black', label='Test Samples')
axes[1, 1].bar(folds, cv_results['train_samples'], color='lightblue', alpha=0.7, 
               edgecolor='black', bottom=0, label='Train Samples')
axes[1, 1].set_ylabel('Muestras', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Fold', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Distribución de Muestras', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/05_metricas_por_fold.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 05_metricas_por_fold.png")
plt.close()

# ============================================================================
# GRÁFICO 6: COMPARACIÓN DE MODELOS
# ============================================================================
if model_comparison is not None:
    print("\n10. Generando comparación de modelos...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics = model_comparison.index
    models = model_comparison.columns
    
    plot_idx = 0
    for idx, metric in enumerate(metrics[:6]):  # Mostrar solo 6 métricas
        ax = axes[idx // 3, idx % 3]
        
        values = model_comparison.loc[metric].values
        colors_bars = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars = ax.bar(models, values, alpha=0.7, edgecolor='black', color=colors_bars)
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} por Modelo', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_comparacion_modelos.png', dpi=300, bbox_inches='tight')
    print("   ✓ Guardado: 06_comparacion_modelos.png")
    plt.close()

# ============================================================================
# GRÁFICO 7: COMPARACIÓN DE MODELOS - ROC MULTIPLE
# ============================================================================
print("\n11. Generando comparación ROC de todos los modelos...")
fig, ax = plt.subplots(figsize=(12, 9))

# ROC del modelo principal
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, 'b-', linewidth=3, label=f'Main Model (AUC = {roc_auc:.4f})', zorder=10)

# ROC de otros modelos
colors_models = plt.cm.tab20(np.linspace(0, 1, len(model_predictions)))
for idx, (model_name, y_pred_model) in enumerate(sorted(model_predictions.items())):
    try:
        fpr_m, tpr_m, _ = roc_curve(y_true, y_pred_model)
        roc_auc_m = auc(fpr_m, tpr_m)
        ax.plot(fpr_m, tpr_m, linewidth=2.5, label=f'{model_name} (AUC = {roc_auc_m:.4f})',
                color=colors_models[idx], alpha=0.8)
    except Exception as e:
        print(f"   Error con {model_name}: {e}")

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Comparación ROC de Todos los Modelos', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/07_roc_comparacion_modelos.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 07_roc_comparacion_modelos.png")
plt.close()

# ============================================================================
# GRÁFICO 8: RESUMEN DE MÉTRICAS PRINCIPALES
# ============================================================================
print("\n12. Generando resumen de métricas...")
from sklearn.metrics import precision_score, recall_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
metrics_values = [accuracy, precision, recall, f1, roc_auc]
colors_metrics = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

bars = ax.bar(metrics_names, metrics_values, color=colors_metrics, alpha=0.7, 
              edgecolor='black', linewidth=2, width=0.6)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Resumen de Métricas Principales (Modelo Principal)', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)

# Añadir valores en las barras
for bar, val in zip(bars, metrics_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/08_resumen_metricas.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 08_resumen_metricas.png")
plt.close()

# ============================================================================
# GRÁFICO 9: ANÁLISIS TEMPORAL
# ============================================================================
print("\n13. Generando análisis temporal...")
try:
    predictions_main['fecha'] = pd.to_datetime(predictions_main['fecha'])
    predictions_main['year_month'] = predictions_main['fecha'].dt.to_period('M')
    
    monthly_accuracy = predictions_main.groupby('year_month').apply(
        lambda x: accuracy_score(x['helada'], (x['probabilidad_helada'] >= 0.5).astype(int))
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    
    # Accuracy mensual
    axes[0].plot(range(len(monthly_accuracy)), monthly_accuracy.values, 
                 marker='o', linewidth=2, markersize=5, color='steelblue')
    axes[0].fill_between(range(len(monthly_accuracy)), monthly_accuracy.values, alpha=0.3)
    axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('Precisión Mensual', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0.7, 1.0])
    
    # Heladas reales vs predichas por mes
    monthly_real = predictions_main.groupby('year_month')['helada'].sum()
    monthly_pred = predictions_main.groupby('year_month').apply(
        lambda x: ((x['probabilidad_helada'] >= 0.5).astype(int)).sum()
    )
    
    x_pos = np.arange(len(monthly_real))
    width = 0.35
    
    axes[1].bar(x_pos - width/2, monthly_real.values, width, label='Real', 
                alpha=0.7, color='coral', edgecolor='black')
    axes[1].bar(x_pos + width/2, monthly_pred.values, width, label='Predicho',
                alpha=0.7, color='skyblue', edgecolor='black')
    
    axes[1].set_ylabel('Número de Heladas', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Mes', fontsize=11, fontweight='bold')
    axes[1].set_title('Heladas Reales vs Predichas (Mensual)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_analisis_temporal.png', dpi=300, bbox_inches='tight')
    print("   ✓ Guardado: 09_analisis_temporal.png")
    plt.close()
except Exception as e:
    print(f"   Error en análisis temporal: {e}")

# ============================================================================
# GRÁFICO 10: DISTRIBUCIÓN DE PROBABILIDADES (VIOLIN PLOT)
# ============================================================================
print("\n14. Generando violin plot...")
fig, ax = plt.subplots(figsize=(10, 6))

data_violin = pd.DataFrame({
    'Probabilidad': list(y_pred_proba[y_true == 0]) + list(y_pred_proba[y_true == 1]),
    'Clase': ['Sin Helada']*sum(y_true == 0) + ['Con Helada']*sum(y_true == 1)
})

sns.violinplot(data=data_violin, x='Clase', y='Probabilidad', ax=ax, palette=['blue', 'red'])
ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold')
ax.set_ylabel('Probabilidad Predicha', fontsize=12, fontweight='bold')
ax.set_xlabel('Clase Real', fontsize=12, fontweight='bold')
ax.set_title('Distribución de Probabilidades por Clase (Violin Plot)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/10_violin_plot.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 10_violin_plot.png")
plt.close()

# ============================================================================
# REPORTE FINAL
# ============================================================================
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS")
print("="*60)
print(f"\nMétricas del Modelo Principal:")
print(f"  - Accuracy:  {accuracy:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall:    {recall:.4f}")
print(f"  - F1-Score:  {f1:.4f}")
print(f"  - AUC-ROC:   {roc_auc:.4f}")

print(f"\nMatriz de Confusión:")
print(f"  - TP (Verdaderos Positivos): {cm[1,1]}")
print(f"  - TN (Verdaderos Negativos): {cm[0,0]}")
print(f"  - FP (Falsos Positivos):     {cm[0,1]}")
print(f"  - FN (Falsos Negativos):     {cm[1,0]}")

print(f"\nValidación Cruzada (5 Folds):")
print(f"  - F1-Score promedio:  {cv_results['f1_score'].mean():.4f} ± {cv_results['f1_score'].std():.4f}")
print(f"  - AUC-ROC promedio:   {cv_results['auc_roc'].mean():.4f} ± {cv_results['auc_roc'].std():.4f}")
print(f"  - Brier Score promedio: {cv_results['brier_score'].mean():.4f} ± {cv_results['brier_score'].std():.4f}")

print(f"\nModelos Evaluados:")
for model_name in sorted(model_predictions.keys()):
    try:
        y_pred_model = model_predictions[model_name]
        acc_m = accuracy_score(y_true, (y_pred_model >= 0.5).astype(int))
        f1_m = f1_score(y_true, (y_pred_model >= 0.5).astype(int))
        fpr_m, tpr_m, _ = roc_curve(y_true, y_pred_model)
        auc_m = auc(fpr_m, tpr_m)
        print(f"  - {model_name:12s}: Accuracy={acc_m:.4f}, F1={f1_m:.4f}, AUC={auc_m:.4f}")
    except ValueError as e:
        print(f"  - {model_name:12s}: Error al calcular métricas - {e}")
    except Exception as e:
        print(f"  - {model_name:12s}: Error inesperado - {e}")

print(f"\n{'='*60}")
print(f"Gráficos guardados en: {os.path.abspath(output_dir)}")
print(f"Total de gráficos generados: 10")
print("="*60)
