"""
sarimax_model.py
Modelo SARIMAX optimizado para prediccion de heladas
Corregido: Sin Data Leakage, ejecucion rapida y sincronizacion temporal
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from utils.data_loading import (
    configure_encoding, load_senamhi_data, prepare_single_station,
)
from utils.evaluation import (
    compute_regression_metrics, sigmoid_probability, print_full_results,
)

configure_encoding()


def train_sarimax():
    print("=" * 70)
    print("MODELO SARIMAX (CORREGIDO Y OPTIMIZADO)")
    print("=" * 70)

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("[ERROR] statsmodels no esta instalado. Instala con: pip install statsmodels")
        return

    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    df = load_senamhi_data()
    station_name, df_est = prepare_single_station(df)
    print(f"Usando estacion: {station_name}")

    # 2. Preparar datos & SOLUCION AL DATA LEAKAGE
    print("[2/4] Generando desfases temporales (Lags)...")
    df_est['precip_ayer'] = df_est['precip'].shift(1)
    df_est['tmax_ayer'] = df_est['tmax'].shift(1)
    df_est['amp_termica_ayer'] = df_est['amp_termica'].shift(1)

    df_est = df_est.dropna()

    y = df_est['tmin']
    exog = df_est[['precip_ayer', 'tmax_ayer', 'amp_termica_ayer']]

    lat_est = df_est['lat'].iloc[0]
    lon_est = df_est['lon'].iloc[0]

    # Division Temporal Sincronizada (Prueba >= 2015)
    df_est['year'] = df_est.index.year
    train_mask = df_est['year'] < 2015
    test_mask = df_est['year'] >= 2015

    y_train, y_test = y[train_mask], y[test_mask]
    exog_train, exog_test = exog[train_mask], exog[test_mask]

    print(f"  Entrenamiento (< 2015): {len(y_train)} dias")
    print(f"  Prueba (>= 2015): {len(y_test)} dias")

    if len(y_test) == 0:
        print("[ERROR] El set de prueba esta vacio. Verifica las fechas.")
        return

    # 3. Entrenar SARIMAX
    print("[3/4] Entrenando SARIMAX en CPU...")
    try:
        model = SARIMAX(
            y_train, exog=exog_train,
            order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False, enforce_invertibility=False,
        )
        results = model.fit(disp=False, maxiter=50)
        print("  Modelo entrenado exitosamente.")
    except Exception as e:
        print(f"[ERROR] Error al entrenar SARIMAX: {e}")
        return

    # 4. Predecir
    print("[4/4] Realizando predicciones sobre matriz de prueba...")
    predictions = results.get_forecast(steps=len(y_test), exog=exog_test)
    y_pred = predictions.predicted_mean
    y_pred.index = y_test.index

    y_test_helada = (y_test.values <= 0).astype(int)
    y_pred_helada = (y_pred.values <= 0).astype(int)

    f1 = f1_score(y_test_helada, y_pred_helada, zero_division=0)
    reg_metrics = compute_regression_metrics(y_test.values, y_pred.values)

    print_full_results("SARIMAX", reg_metrics,
                       {'f1': f1, 'precision': 0, 'recall': 0, 'tss': 0})

    # 5. Guardar predicciones
    resultados = pd.DataFrame({
        'fecha': y_test.index.strftime('%Y-%m-%d'),
        'tmin_real': y_test.values,
        'tmin_pred': y_pred.values,
        'helada_real': y_test_helada,
        'helada_pred': y_pred_helada,
        'prob_helada_sarimax': sigmoid_probability(y_pred.values),
    })
    resultados['lat'] = round(lat_est, 2)
    resultados['lon'] = round(lon_est, 2)

    resultados.to_csv('data_process/predictions_sarimax.csv', index=False)
    print(f"\n[OK] Predicciones SARIMAX guardadas en: data_process/predictions_sarimax.csv")


if __name__ == '__main__':
    train_sarimax()
