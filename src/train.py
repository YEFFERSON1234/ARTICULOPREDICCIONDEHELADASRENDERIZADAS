import os
import pandas as pd
import joblib
import xgboost as xgb


def prepare_features(df):
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(['estacion', 'fecha']).reset_index(drop=True)
    df['day_of_year'] = df['fecha'].dt.dayofyear
    df['month'] = df['fecha'].dt.month
    df['year'] = df['fecha'].dt.year
    for lag in [1, 2, 3]:
        df[f'tmin_lag_{lag}'] = df.groupby('estacion')['tmin'].shift(lag)
    df = df.dropna()
    return df


def train_models(csv_path, models_dir='models'):
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = prepare_features(df)

    feature_cols = ['lat', 'lon', 'day_of_year', 'month', 'precip', 'tmax',
                    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3']
    X = df[feature_cols]
    y_reg = df['tmin']
    y_clf = (df['tmin'] <= 0).astype(int)

    ultimo_anio = df['year'].max()
    train_mask = df['year'] < ultimo_anio

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
        csv_path = os.path.join('limpiezadedatos', 'datos_heladas_puno_REAL.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró {csv_path}")
    train_models(csv_path)
