"""
evaluation.py
Shared utilities for model evaluation, metrics computation, and result export.

Consolidates duplicated patterns from:
  - modelos/{xgboost_model, random_forest, mlp_model, cnn_1d_model, ...}.py
  - modelos/{RandomForest,SVM,XGBoost}/entrenamiento_*.py
  - src/evaluate_models.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_regression_metrics(y_true, y_pred):
    """Compute RMSE between true and predicted values.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    dict with ``rmse``.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {'rmse': rmse}


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """Compute standard classification metrics for frost prediction.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth.
    y_pred : array-like
        Binary predictions.
    y_prob : array-like or None
        Predicted probabilities for the positive class.

    Returns
    -------
    dict with ``f1``, ``precision``, ``recall``, ``tss``, and optionally ``auc_roc``.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    tss = (tp / (tp + fn) if (tp + fn) > 0 else 0.0) + \
          (tn / (tn + fp) if (tn + fp) > 0 else 0.0) - 1

    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tss': tss,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }

    if y_prob is not None:
        try:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics['auc_roc'] = 0.0

    return metrics


def print_regression_results(model_name, metrics):
    """Print RMSE result in the standard project format.

    Parameters
    ----------
    model_name : str
    metrics : dict
        Must contain ``rmse``.
    """
    print(f"\n{'=' * 70}")
    print(f"RESULTADOS {model_name.upper()}")
    print(f"{'=' * 70}")
    print(f"RMSE (temperatura): {metrics['rmse']:.4f} C")


def print_classification_results(model_name, metrics):
    """Print classification metrics in the standard project format.

    Parameters
    ----------
    model_name : str
    metrics : dict
        Output of :func:`compute_classification_metrics`.
    """
    print(f"\n{'=' * 70}")
    print(f"RESULTADOS {model_name.upper()}")
    print(f"{'=' * 70}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    if 'auc_roc' in metrics:
        print(f"AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"TSS:       {metrics['tss']:.4f}")


def print_full_results(model_name, reg_metrics=None, clf_metrics=None):
    """Print both regression and classification metrics.

    Parameters
    ----------
    model_name : str
    reg_metrics : dict or None
    clf_metrics : dict or None
    """
    print(f"\n{'=' * 70}")
    print(f"RESULTADOS {model_name.upper()}")
    print(f"{'=' * 70}")
    if reg_metrics:
        print(f"RMSE (temperatura): {reg_metrics['rmse']:.4f} C")
    if clf_metrics:
        print(f"F1-Score: {clf_metrics['f1']:.4f}")
        if 'auc_roc' in clf_metrics:
            print(f"AUC-ROC:  {clf_metrics['auc_roc']:.4f}")


def print_table_metrics(model_name, metrics):
    """Print metrics formatted for the article's Table II.

    Used by ``modelos/{RandomForest,SVM,XGBoost}/entrenamiento_*.py``.

    Parameters
    ----------
    model_name : str
    metrics : dict
        Output of :func:`compute_classification_metrics`.
    """
    print("\n" + "=" * 45)
    print("=== EXTRAE ESTOS DATOS PARA TU TABLA II ===")
    print("=" * 45)
    print(f"Modelo: {model_name}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-Score:  {metrics['f1']:.3f}")
    print(f"TSS:       {metrics['tss']:.3f}")
    print("=" * 45 + "\n")


def sigmoid_probability(y_pred):
    """Convert raw temperature predictions to frost probability via sigmoid.

    Maps predictions through ``1 / (1 + exp(y_pred))`` so that lower (more
    negative) temperatures yield higher frost probabilities.

    Parameters
    ----------
    y_pred : array-like
        Predicted temperature values.

    Returns
    -------
    np.ndarray
        Probabilities clipped to ``[0, 1]``.
    """
    return np.clip(1 / (1 + np.exp(np.asarray(y_pred, dtype=float))), 0, 1)


def save_predictions(df_test, output_path, extra_cols=None, normalize_keys=True):
    """Save a predictions DataFrame to CSV with optional key normalization.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test-set slice (copy) with prediction columns already attached.
    output_path : str
        Destination CSV path.
    extra_cols : dict[str, array-like] or None
        Additional columns to attach before saving.
    normalize_keys : bool
        If ``True``, round ``lat``/``lon`` to 2 decimals and format ``fecha``
        as ``%Y-%m-%d``.
    """
    result = df_test.copy()

    if extra_cols:
        for col_name, values in extra_cols.items():
            result[col_name] = values

    if normalize_keys:
        if 'fecha' in result.columns:
            result['fecha'] = pd.to_datetime(result['fecha']).dt.strftime('%Y-%m-%d')
        if 'lat' in result.columns:
            result['lat'] = result['lat'].round(2)
        if 'lon' in result.columns:
            result['lon'] = result['lon'].round(2)

    import os
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[OK] Predicciones guardadas en: {output_path}")
