import os
import pandas as pd
import joblib


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


def predict(csv_path, models_dir='models', out_path='data_process/predictions_pipeline.csv'):
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"No se encontró carpeta de modelos: {models_dir}")
    reg = joblib.load(os.path.join(models_dir, 'xgb_reg.pkl'))
    clf = joblib.load(os.path.join(models_dir, 'xgb_clf.pkl'))

    df = pd.read_csv(csv_path)
    df_pre = prepare_features(df)

    feature_cols = ['lat', 'lon', 'day_of_year', 'month', 'precip', 'tmax',
                    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3']
    X = df_pre[feature_cols]

    ultimo_anio = df_pre['year'].max()
    test_mask = df_pre['year'] >= ultimo_anio
    X_test = X[test_mask]

    preds = reg.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    resultados = df_pre[test_mask].copy()
    resultados['tmin_pred'] = preds
    resultados['probabilidad_helada'] = probs
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resultados.to_csv(out_path, index=False)
    print(f"Predicciones guardadas en {out_path}")


if __name__ == '__main__':
    csv_path = os.path.join('data_process', 'datos_heladas_puno_REAL.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join('data_process', 'dataset_ML_final_completo.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró {csv_path}")
    predict(csv_path)
