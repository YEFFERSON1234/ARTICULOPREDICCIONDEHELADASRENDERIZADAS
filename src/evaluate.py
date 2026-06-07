import os
import pandas as pd
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, roc_auc_score
import numpy as np


def evaluate(predictions_csv, out_dir='graficos_resultados'):
    df = pd.read_csv(predictions_csv)
    if 'tmin' not in df.columns or 'tmin_pred' not in df.columns:
        raise ValueError('El CSV de predicciones debe contener `tmin` y `tmin_pred`.')
    y_true = df['tmin'].values
    y_pred = df['tmin_pred'].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # clasificación de helada
    y_frost = (df['tmin'] <= 0).astype(int)
    y_frost_pred = (df['tmin_pred'] <= 0).astype(int)

    report = classification_report(y_frost, y_frost_pred, output_dict=True)
    cm = confusion_matrix(y_frost, y_frost_pred)

    roc = None
    if 'probabilidad_helada' in df.columns:
        try:
            roc = roc_auc_score(y_frost, df['probabilidad_helada'].values)
        except ValueError as e:
            print(f"[WARNING] No se pudo calcular AUC-ROC: {e}")
            roc = None

    os.makedirs(out_dir, exist_ok=True)
    summary = {
        'rmse': float(rmse),
        'roc_auc': float(roc) if roc is not None else None,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    out_path = os.path.join(out_dir, 'evaluation_summary.json')
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Evaluación guardada en {out_path}")


if __name__ == '__main__':
    pred_csv = os.path.join('data_process', 'predictions_pipeline.csv')
    if not os.path.exists(pred_csv):
        pred_csv = os.path.join('limpiezadedatos', 'predictions.csv')
    if not os.path.exists(pred_csv):
        raise FileNotFoundError('No se encontró archivo de predicciones para evaluar.')
    evaluate(pred_csv)
