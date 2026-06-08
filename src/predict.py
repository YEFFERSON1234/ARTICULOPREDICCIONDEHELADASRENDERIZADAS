import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from utils.data_loading import prepare_features, DEFAULT_FEATURE_COLS, temporal_train_test_split
from utils.evaluation import save_predictions


def predict(csv_path, models_dir='models', out_path='data_process/predictions_pipeline.csv'):
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"No se encontro carpeta de modelos: {models_dir}")
    reg = joblib.load(os.path.join(models_dir, 'xgb_reg.pkl'))
    clf = joblib.load(os.path.join(models_dir, 'xgb_clf.pkl'))

    df = pd.read_csv(csv_path)
    df_pre = prepare_features(df)

    X = df_pre[DEFAULT_FEATURE_COLS]

    _, test_mask = temporal_train_test_split(df_pre)
    X_test = X[test_mask]

    preds = reg.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    save_predictions(
        df_pre[test_mask],
        out_path,
        extra_cols={
            'tmin_pred': preds,
            'probabilidad_helada': probs,
        },
        normalize_keys=False,
    )


if __name__ == '__main__':
    csv_path = os.path.join('data_process', 'datos_heladas_puno_REAL.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join('data_process', 'dataset_ML_final_completo.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontro {csv_path}")
    predict(csv_path)
