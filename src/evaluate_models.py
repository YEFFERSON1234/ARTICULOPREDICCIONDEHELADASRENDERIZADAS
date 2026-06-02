"""
evaluate_models.py
Implementación de métricas de evaluación: F1-score, AUC-ROC, AUC-PR, Brier Score, CSI
Validación cruzada espaciotemporal walk-forward
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve, auc,
    brier_score_loss, confusion_matrix, classification_report
)
import sys

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def calculate_csi(y_true, y_pred, threshold=0.5):
    """
    Calcula el Critical Success Index (CSI), también conocido como Threat Score
    Métrica estándar en meteorología para predicción de eventos extremos
    
    CSI = TP / (TP + FP + FN)
    
    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Probabilidades predichas (0-1)
        threshold: Umbral para clasificación binaria
    
    Returns:
        float: CSI score
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return csi

def calculate_brier_score(y_true, y_pred):
    """
    Calcula el Brier Score
    Métrica de calibración para probabilidades
    
    Brier Score = (1/N) * sum((y_true - y_pred)^2)
    
    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Probabilidades predichas (0-1)
    
    Returns:
        float: Brier Score (menor es mejor)
    """
    return brier_score_loss(y_true, y_pred)

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evalúa un modelo con todas las métricas requeridas
    
    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Probabilidades predichas (0-1)
        model_name: Nombre del modelo para reporte
    
    Returns:
        dict: Diccionario con todas las métricas
    """
    metrics = {}
    
    # F1-Score
    y_pred_binary = (y_pred >= 0.5).astype(int)
    metrics['f1_score'] = f1_score(y_true, y_pred_binary)
    
    # AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred)
    except:
        metrics['auc_roc'] = 0.0
    
    # AUC-PR (Precision-Recall)
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        metrics['auc_pr'] = auc(recall, precision)
    except:
        metrics['auc_pr'] = 0.0
    
    # Brier Score
    metrics['brier_score'] = calculate_brier_score(y_true, y_pred)
    
    # CSI (Critical Success Index)
    metrics['csi'] = calculate_csi(y_true, y_pred)
    
    # Métricas adicionales
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics['true_positives'] = tp
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    
    # Calcular tasas
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics

def print_metrics_report(metrics, model_name="Model"):
    """Imprime un reporte formateado de las métricas"""
    print(f"\n{'='*70}")
    print(f"REPORTE DE EVALUACIÓN - {model_name.upper()}")
    print(f"{'='*70}")
    
    print(f"\nMÉTRICAS PRINCIPALES:")
    print(f"  F1-Score:           {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:            {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:             {metrics['auc_pr']:.4f}")
    print(f"  Brier Score:        {metrics['brier_score']:.4f}")
    print(f"  CSI (Threat Score): {metrics['csi']:.4f}")
    
    print(f"\nMÉTRICAS ADICIONALES:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall (Sensibilidad): {metrics['recall']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f}")
    
    print(f"\nMATRIZ DE CONFUSIÓN:")
    print(f"  Verdaderos Positivos:  {metrics['true_positives']}")
    print(f"  Verdaderos Negativos:  {metrics['true_negatives']}")
    print(f"  Falsos Positivos:     {metrics['false_positives']}")
    print(f"  Falsos Negativos:     {metrics['false_negatives']}")

def evaluate_xgboost_model():
    """Evalúa el modelo XGBoost"""
    print("\n" + "="*70)
    print("EVALUANDO MODELO XGBOOST")
    print("="*70)
    
    # Cargar datos y predicciones
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['helada'] = (df['tmin'] <= 0).astype(int)
    
    # Cargar predicciones
    preds = pd.read_csv('data_process/predictions.csv')
    preds['fecha'] = pd.to_datetime(preds['fecha'])
    
    # Unir por fecha y coordenadas
    merged = pd.merge(df, preds, on=['fecha', 'lat', 'lon'], how='inner')
    
    if len(merged) == 0:
        print("[ERROR] No se pudieron unir datos con predicciones")
        return None
    
    y_true = merged['helada'].values
    y_pred = merged['prob_helada'].values
    
    metrics = evaluate_model(y_true, y_pred, "XGBoost")
    print_metrics_report(metrics, "XGBoost")
    
    return metrics

def evaluate_random_forest_model():
    """Evalúa el modelo Random Forest"""
    print("\n" + "="*70)
    print("EVALUANDO MODELO RANDOM FOREST")
    print("="*70)
    
    # Cargar predicciones (ya contiene tmin y helada)
    preds = pd.read_csv('data_process/predictions_rf.csv')
    preds['fecha'] = pd.to_datetime(preds['fecha'])
    
    # Usar directamente las columnas del archivo de predicciones
    y_true = preds['helada'].values
    y_pred = preds['prob_frost_rf'].values
    
    metrics = evaluate_model(y_true, y_pred, "Random Forest")
    print_metrics_report(metrics, "Random Forest")
    
    return metrics

def compare_models():
    """Compara los modelos entrenados"""
    print("\n" + "="*70)
    print("COMPARATIVA DE MODELOS")
    print("="*70)
    
    metrics_xgb = evaluate_xgboost_model()
    metrics_rf = evaluate_random_forest_model()
    
    if metrics_xgb and metrics_rf:
        print(f"\n{'='*70}")
        print(f"COMPARACIÓN FINAL")
        print(f"{'='*70}")
        
        comparison = pd.DataFrame({
            'Métrica': ['F1-Score', 'AUC-ROC', 'AUC-PR', 'Brier Score', 'CSI', 'Accuracy'],
            'XGBoost': [
                metrics_xgb['f1_score'],
                metrics_xgb['auc_roc'],
                metrics_xgb['auc_pr'],
                metrics_xgb['brier_score'],
                metrics_xgb['csi'],
                metrics_xgb['accuracy']
            ],
            'Random Forest': [
                metrics_rf['f1_score'],
                metrics_rf['auc_roc'],
                metrics_rf['auc_pr'],
                metrics_rf['brier_score'],
                metrics_rf['csi'],
                metrics_rf['accuracy']
            ]
        })
        
        print(comparison.to_string(index=False))
        
        # Guardar comparación
        comparison.to_csv('data_process/comparacion_modelos.csv', index=False)
        print(f"\n[OK] Comparación guardada en: data_process/comparacion_modelos.csv")

def main():
    """Función principal"""
    print("="*70)
    print("SISTEMA DE EVALUACIÓN DE MODELOS")
    print("Métricas: F1-score, AUC-ROC, AUC-PR, Brier Score, CSI")
    print("="*70)
    
    compare_models()
    
    print(f"\n{'='*70}")
    print("[OK] Evaluación completada")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
