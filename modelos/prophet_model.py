"""
prophet_model.py
Modelo Prophet de Facebook para prediccion de heladas
Corregido: Error de indice solucionado, sin Data Leakage y sincronizado.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from utils.data_loading import (
    configure_encoding, load_senamhi_data, prepare_single_station,
    temporal_train_test_split,
)
from utils.evaluation import (
    compute_regression_metrics, sigmoid_probability, print_full_results,
)

configure_encoding()


def train_prophet():
    print("=" * 70)
    print("MODELO PROPHET (CORREGIDO)")
    print("=" * 70)

    try:
        from prophet import Prophet
    except ImportError:
        print("[ERROR] Prophet no esta instalado. Instala con: pip install prophet")
        return

    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    df = load_senamhi_data()

    station_name = df['estacion'].value_counts().index[0]
    print(f"Usando estacion: {station_name}")
    df_est = df[df['estacion'] == station_name].copy()

    # SOLUCION AL KEYERROR: Conservar 'fecha' explicitamente junto a las numericas
    numeric_cols = list(df_est.select_dtypes(include=[np.number]).columns)
    if 'fecha' not in numeric_cols:
        cols_a_guardar = ['fecha'] + numeric_cols

    df_est = df_est[cols_a_guardar]

    # Resample diario limpio
    df_est = df_est.set_index('fecha')
    df_est = df_est.resample('D').mean()
    df_est = df_est.ffill()
    df_est = df_est.reset_index()

    # 2. Preparar datos para Prophet & SOLUCION AL DATA LEAKAGE
    print("[2/4] Preparando desfases temporales y estructuras...")
    df_prophet = pd.DataFrame()
    df_prophet['ds'] = df_est['fecha']
    df_prophet['y'] = df_est['tmin']

    df_prophet['precip_ayer'] = df_est['precip'].shift(1)
    df_prophet['tmax_ayer'] = df_est['tmax'].shift(1)
    df_prophet['amp_termica_ayer'] = df_est['amp_termica'].shift(1)

    lat_est = df_est['lat'].iloc[0]
    lon_est = df_est['lon'].iloc[0]

    df_prophet = df_prophet.dropna().reset_index(drop=True)

    # Dividir datos (Prueba >= 2015)
    df_prophet['year'] = df_prophet['ds'].dt.year
    train_mask, test_mask = temporal_train_test_split(df_prophet, test_year=2015)
    train_df = df_prophet[train_mask].copy()
    test_df = df_prophet[test_mask].copy()

    print(f"  Entrenamiento ( < 2015): {len(train_df)} dias")
    print(f"  Prueba (>= 2015): {len(test_df)} dias")

    if len(test_df) == 0:
        print("[ERROR] El set de prueba quedo vacio. Revisa las fechas del archivo original.")
        return

    # 3. Entrenar Prophet
    print("[3/4] Entrenando Prophet con regresores historicos...")
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
        )

        model.add_regressor('precip_ayer')
        model.add_regressor('tmax_ayer')
        model.add_regressor('amp_termica_ayer')

        model.fit(train_df)
        print("Modelo Prophet entrenado exitosamente.")
    except Exception as e:
        print(f"[ERROR] No se pudo entrenar: {e}")
        return

    # 4. Predecir
    print("[4/4] Realizando predicciones sobre matriz de prueba...")
    future = pd.concat([train_df, test_df], ignore_index=True)[
        ['ds', 'precip_ayer', 'tmax_ayer', 'amp_termica_ayer']
    ]
    forecast = model.predict(future)

    y_pred = forecast['yhat'].iloc[-len(test_df):].values
    y_test = test_df['y'].values

    y_test_helada = (y_test <= 0).astype(int)
    y_pred_helada = (y_pred <= 0).astype(int)

    f1 = f1_score(y_test_helada, y_pred_helada, zero_division=0)
    reg_metrics = compute_regression_metrics(y_test, y_pred)

    print_full_results("PROPHET", reg_metrics,
                       {'f1': f1, 'precision': 0, 'recall': 0, 'tss': 0})

    # 5. Guardar predicciones
    resultados = pd.DataFrame({
        'fecha': test_df['ds'].dt.strftime('%Y-%m-%d'),
        'tmin_real': y_test,
        'tmin_pred': y_pred,
        'helada_real': y_test_helada,
        'helada_pred': y_pred_helada,
        'prob_helada_prophet': sigmoid_probability(y_pred),
    })
    resultados['lat'] = round(lat_est, 2)
    resultados['lon'] = round(lon_est, 2)

    resultados.to_csv('data_process/predictions_prophet.csv', index=False)
    print(f"\n[OK] Predicciones Prophet guardadas en: data_process/predictions_prophet.csv")


if __name__ == '__main__':
    train_prophet()
