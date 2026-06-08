"""
plotting.py
Shared plotting utilities for confusion matrices and ROC curves.

Consolidates near-identical plotting code from:
  - modelos/RandomForest/entrenamiento_rf.py
  - modelos/SVM/entrenamiento_svm.py
  - modelos/XGBoost/entrenamiento_xgb.py
  - modelos/SVM.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(y_true, y_pred, model_name, save_path,
                          cmap='Blues', labels=None):
    """Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    model_name : str
        Model name shown in the title.
    save_path : str
        Full file path to save the PNG.
    cmap : str
        Matplotlib/seaborn colormap name.
    labels : list[str] or None
        Tick labels. Defaults to ``['No Helada', 'Helada']``.
    """
    if labels is None:
        labels = ['No Helada', 'Helada']

    cm = confusion_matrix(y_true, y_pred)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Matriz de Confusi\u00f3n - {model_name}')
    plt.ylabel('Realidad (SENAMHI)')
    plt.xlabel('Predicci\u00f3n (Modelo)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, model_name, save_path,
                   color='darkorange'):
    """Plot and save a ROC curve.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    model_name : str
        Model name shown in the title / legend.
    save_path : str
        Full file path to save the PNG.
    color : str
        Line color for the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - Detecci\u00f3n de Heladas ({model_name})')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
