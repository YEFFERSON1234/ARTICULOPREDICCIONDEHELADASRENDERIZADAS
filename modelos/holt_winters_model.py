"""
holt_winters_model.py
Modelo Holt-Winters (Exponential Smoothing) para prediccion de heladas
Modelo estadistico tradicional para series temporales
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


def train_holt_winters():
    """Entrena modelo Holt-Winters para prediccion de heladas de forma correcta"""
    print("=" * 70)
    print("MODELO HOLT-WINTERS (CORREGIDO)")
    print("=" * 70)

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        print("[ERROR] statsmodels no esta instalado. Instala con: pip install statsmodels")
        return

    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    df = load_senamhi_data()
    station_name, df_est = prepare_single_station(df)
    print(f"Usando estacion: {station_name}")

    # 2. Preparar datos
    print("[2/4] Preparando datos...")
    y = df_est['tmin']

    train_size = int(len(y) * 0.8)
    y_train = y[:train_size]
    y_test = y[train_size:]

    print(f"Entrenamiento: {len(y_train)} dias")
    print(f"Prueba: {len(y_test)} dias")

    # 3. Entrenar Holt-Winters
    print("[3/4] Entrenando Holt-Winters...")
    try:
        model = ExponentialSmoothing(
            y_train, trend='add', seasonal='add',
            seasonal_periods=365, damped_trend=True,
        )
        results = model.fit(optimized=True, use_brute=True)
        print("Modelo Holt-Winters avanzado entrenado exitosamente")
    except Exception as e:
        print(f"[WARN] No se pudo entrenar con estacionalidad anual: {e}")
        print("Usando version simplificada...")
        model = ExponentialSmoothing(y_train, trend='add', seasonal=None, damped_trend=True)
        results = model.fit()
        print("Modelo simplificado entrenado")

    # 4. Predecir
    print("[4/4] Realizando predicciones...")
    y_pred = results.forecast(steps=len(y_test))
    y_pred.index = y_test.index

    y_test_helada = (y_test <= 0).astype(int)
    y_pred_helada = (y_pred <= 0).astype(int)

    f1 = f1_score(y_test_helada, y_pred_helada, zero_division=0)
    reg_metrics = compute_regression_metrics(y_test, y_pred)

    print_full_results("HOLT-WINTERS", reg_metrics,
                       {'f1': f1, 'precision': 0, 'recall': 0, 'tss': 0})

    # Guardar predicciones alineadas
    resultados = pd.DataFrame({
        'fecha': y_test.index.strftime('%Y-%m-%d'),
        'tmin_real': y_test.values,
        'tmin_pred': y_pred.values,
        'helada_real': y_test_helada.values,
        'helada_pred': y_pred_helada.values,
        'prob_frost_hw': sigmoid_probability(y_pred.values),
    })

    resultados['lat'] = df_est['lat'].iloc[0]
    resultados['lon'] = df_est['lon'].iloc[0]

    resultados.to_csv('data_process/predictions_holt_winters.csv', index=False)
    print(f"\n[OK] Predicciones Holt-Winters guardadas en: data_process/predictions_holt_winters.csv")


if __name__ == '__main__':
    train_holt_winters()
