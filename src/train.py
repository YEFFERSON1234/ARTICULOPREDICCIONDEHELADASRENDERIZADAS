import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
import xgboost as xgb
from utils.data_loading import prepare_features, DEFAULT_FEATURE_COLS, temporal_train_test_split


def train_models(csv_path, models_dir='models'):
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = prepare_features(df)

    X = df[DEFAULT_FEATURE_COLS]
    y_reg = df['tmin']
    y_clf = (df['tmin'] <= 0).astype(int)

    train_mask, _ = temporal_train_test_split(df)
    X_train = X[train_mask]
    y_reg_train = y_reg[train_mask]
    y_clf_train = y_clf[train_mask]

    print(f"Entrenando XGBoost regressor con {len(X_train)} filas...")
    reg = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.04, tree_method='hist')
    reg.fit(X_train, y_reg_train)

    print("Entrenando XGBoost classifier...")
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.04, tree_method='hist')
    clf.fit(X_train, y_clf_train)

    joblib.dump(reg, os.path.join(models_dir, 'xgb_reg.pkl'))
    joblib.dump(clf, os.path.join(models_dir, 'xgb_clf.pkl'))
    print(f"Modelos guardados en {models_dir}")


if __name__ == '__main__':
    csv_path = os.path.join('data_process', 'datos_heladas_puno_REAL.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontro {csv_path}")
    train_models(csv_path)
