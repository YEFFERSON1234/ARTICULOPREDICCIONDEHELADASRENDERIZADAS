"""
sarima_ann_hybrid.py
Modelo Hibrido SARIMA+ANN (Artificial Neural Network)
Corregido: Sin Data Leakage, indexacion limpia y correccion de tipos NumPy
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score, mean_squared_error
from utils.data_loading import (
    configure_encoding, load_senamhi_data, prepare_single_station,
    scale_features,
)
from utils.evaluation import (
    compute_regression_metrics, sigmoid_probability, print_full_results,
)

configure_encoding()


def train_sarima_ann_hybrid():
    print("=" * 70)
    print("MODELO HIBRIDO SARIMA+ANN (CORREGIDO)")
    print("=" * 70)

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("[ERROR] statsmodels no esta instalado. Instala con: pip install statsmodels")
        return

    # 1. Cargar datos
    print("\n[1/5] Cargando datos...")
    df = load_senamhi_data()
    station_name, df_est = prepare_single_station(df)
    print(f"Usando estacion: {station_name}")

    # 2. Preparar datos & SOLUCION AL DATA LEAKAGE
    print("[2/5] Generando desfases temporales (Lags)...")
    y = df_est['tmin']

    df_est['precip_ayer'] = df_est['precip'].shift(1)
    df_est['tmax_ayer'] = df_est['tmax'].shift(1)
    df_est['amp_termica_ayer'] = df_est['amp_termica'].shift(1)

    df_est = df_est.dropna()
    y = df_est['tmin']
    exog = df_est[['precip_ayer', 'tmax_ayer', 'amp_termica_ayer']]

    # Division Temporal Sincronizada (Prueba >= 2015)
    df_est['year'] = df_est.index.year
    train_mask = df_est['year'] < 2015
    test_mask = df_est['year'] >= 2015

    y_train, y_test = y[train_mask], y[test_mask]
    exog_train, exog_test = exog[train_mask], exog[test_mask]

    print(f"  Entrenamiento (< 2015): {len(y_train)} dias")
    print(f"  Prueba (>= 2015): {len(y_test)} dias")

    if len(y_test) == 0:
        print("[ERROR] El set de prueba esta vacio. Verifica los anios disponibles.")
        return

    # 3. Entrenar Componente Lineal (SARIMA)
    print("[3/5] Entrenando componente lineal SARIMA...")
    try:
        sarima_model = SARIMAX(
            y_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False, enforce_invertibility=False,
        )
        sarima_results = sarima_model.fit(disp=False, maxiter=50)

        sarima_pred = sarima_results.get_forecast(steps=len(y_test)).predicted_mean
        sarima_pred.index = y_test.index
        print("  SARIMA entrenado con exito.")
    except Exception as e:
        print(f"[ERROR] Error critico en SARIMA: {e}")
        return

    # 4. Calcular Residuos y Entrenar Componente No Lineal (ANN)
    print("[4/5] Entrenando componente ANN para modelar residuos...")
    sarima_train_pred = sarima_results.predict(start=y_train.index[0], end=y_train.index[-1])
    residuals_train = y_train.values - sarima_train_pred.values

    exog_train_scaled, exog_test_scaled, _ = scale_features(exog_train, exog_test)

    ann_model = MLPRegressor(
        hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
        alpha=0.005, max_iter=150, random_state=42,
        early_stopping=True, validation_fraction=0.1,
    )
    ann_model.fit(exog_train_scaled, residuals_train)
    print("  ANN entrenada con exito.")

    # 5. Prediccion Hibrida Combinada
    print("[5/5] Realizando predicciones hibridas finales...")
    residuals_pred = ann_model.predict(exog_test_scaled)
    y_pred = sarima_pred.values + residuals_pred

    y_test_helada = (y_test.values <= 0).astype(int)
    y_pred_helada = (y_pred <= 0).astype(int)

    f1 = f1_score(y_test_helada, y_pred_helada, zero_division=0)
    reg_metrics = compute_regression_metrics(y_test.values, y_pred)

    print_full_results("MODELO HIBRIDO SARIMA+ANN", reg_metrics,
                       {'f1': f1, 'precision': 0, 'recall': 0, 'tss': 0})

    # 6. Exportar Predicciones Alineadas
    resultados = pd.DataFrame({
        'fecha': y_test.index.strftime('%Y-%m-%d'),
        'tmin_real': y_test.values,
        'tmin_pred': y_pred,
        'tmin_sarima': sarima_pred.values,
        'residual_ann': residuals_pred,
        'helada_real': y_test_helada,
        'helada_pred': y_pred_helada,
        'prob_helada_hybrid': sigmoid_probability(y_pred),
    })

    resultados['lat'] = round(df_est['lat'].iloc[0], 2)
    resultados['lon'] = round(df_est['lon'].iloc[0], 2)

    resultados.to_csv('data_process/predictions_sarima_ann_hybrid.csv', index=False)
    print(f"\n[OK] Predicciones guardadas: data_process/predictions_sarima_ann_hybrid.csv")


if __name__ == '__main__':
    train_sarima_ann_hybrid()
