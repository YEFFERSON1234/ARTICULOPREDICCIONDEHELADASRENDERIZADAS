"""
generate_missing_plots.py
Genera gráficos estándar para modelos que no tienen imágenes completas
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def generate_model_plots(model_name, predictions_csv, output_dir):
    """
    Genera los tres gráficos estándar para un modelo:
    1. Curva ROC
    2. Matriz de confusión
    3. Gráfico de dispersión
    """
    print(f"\nGenerando gráficos para {model_name}...")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos
    try:
        df = pd.read_csv(predictions_csv)
        print(f"  - Cargados {len(df)} registros desde {predictions_csv}")
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar {predictions_csv}: {e}")
        return
    
    # Verificar columnas necesarias
    required_cols = ['helada', 'prob_helada']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"  [ERROR] Faltan columnas: {missing_cols}")
        print(f"  Columnas disponibles: {df.columns.tolist()}")
        return
    
    # 1. Curva ROC
    try:
        y_true = df['helada'].values
        y_scores = df['prob_helada'].values
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        roc_path = os.path.join(output_dir, 'curva_roc.png')
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Curva ROC guardada en {roc_path}")
    except Exception as e:
        print(f"  [ERROR] No se pudo generar curva ROC: {e}")
    
    # 2. Matriz de confusión
    try:
        y_pred_binary = (y_scores >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Helada', 'Helada'],
                   yticklabels=['No Helada', 'Helada'])
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'matriz_confusion.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Matriz de confusión guardada en {cm_path}")
    except Exception as e:
        print(f"  [ERROR] No se pudo generar matriz de confusión: {e}")
    
    # 3. Gráfico de dispersión
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_scores, y_true, alpha=0.3, s=10)
        plt.axvline(x=0.5, color='r', linestyle='--', label='Umbral 0.5')
        plt.xlabel('Probabilidad Predicha de Helada')
        plt.ylabel('Helada Real (0=No, 1=Sí)')
        plt.title(f'Gráfico de Dispersión - {model_name}')
        plt.legend()
        plt.tight_layout()
        
        scatter_path = os.path.join(output_dir, 'grafico_dispersion.png')
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Gráfico de dispersión guardado en {scatter_path}")
    except Exception as e:
        print(f"  [ERROR] No se pudo generar gráfico de dispersión: {e}")
    
    print(f"  [OK] Gráficos generados para {model_name}")

def main():
    """Función principal"""
    print("="*70)
    print("GENERADOR DE GRÁFICOS PARA MODELOS FALTANTES")
    print("="*70)
    
    # Definir modelos y sus rutas
    models_config = [
        {
            'name': 'MLP',
            'predictions_csv': 'data_process/predictions_mlp.csv',
            'output_dir': 'modelos/MLP/graficos_resultados'
        },
        {
            'name': 'Ensemble',
            'predictions_csv': 'data_process/predictions_ensemble.csv',
            'output_dir': 'modelos/Ensemble/graficos_resultados'
        }
    ]
    
    for model in models_config:
        if os.path.exists(model['predictions_csv']):
            generate_model_plots(
                model['name'],
                model['predictions_csv'],
                model['output_dir']
            )
        else:
            print(f"\n[AVISO] No se encontró {model['predictions_csv']}")
            print(f"  Saltando generación de gráficos para {model['name']}")
    
    print(f"\n{'='*70}")
    print("[OK] Proceso completado")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
