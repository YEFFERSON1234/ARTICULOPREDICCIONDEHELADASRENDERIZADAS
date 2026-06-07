"""
ensemble_models.py
Modulo de unificacion y ensamble ponderado para prediccion regional de heladas
Integra: XGBoost, Random Forest, LSTM y SVM (Validacion Multiestacion)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from utils.data_loading import configure_encoding

configure_encoding()


def ensamblar_con_diagnostico():
    print("=" * 70)
    print("INICIANDO ENSAMBLE DE MODELOS MULTIESTACION - PUNO")
    print("=" * 70)

    # 1. CARGA DINAMICA DE PREDICCIONES
    modelos_disponibles = {}

    try:
        df_xgb = pd.read_csv('data_process/predictions.csv')
        modelos_disponibles['xgb'] = df_xgb
        print("[OK] Predicciones de XGBoost cargadas con exito.")
    except FileNotFoundError:
        print("[ERROR] Archivo 'predictions.csv' (XGBoost) no encontrado.")
        return

    try:
        df_rf = pd.read_csv('data_process/predictions_rf.csv')
        modelos_disponibles['rf'] = df_rf
        print("[OK] Predicciones de Random Forest cargadas con exito.")
    except FileNotFoundError:
        print("[ERROR] Archivo 'predictions_rf.csv' (Random Forest) no encontrado.")
        return

    try:
        df_lstm = pd.read_csv('data_process/predictions_lstm.csv')
        modelos_disponibles['lstm'] = df_lstm
        print("[INFO] LSTM detectado y disponible para el ensamble.")
    except FileNotFoundError:
        print("[INFO] LSTM no detectado. Se omitira en la combinacion.")

    try:
        df_svm = pd.read_csv('data_process/predictions_svm.csv')
        modelos_disponibles['svm'] = df_svm
        print("[INFO] SVM Calibrado detectado y disponible para el ensamble.")
    except FileNotFoundError:
        print("[INFO] SVM no detectado. Se omitira en la combinacion.")

    # 2. SELECCION DE COLUMNA DE PROBABILIDAD DE XGBOOST
    col_xgb = None
    for col in ['probabilidad_helada', 'prob_helada', 'probabilidad']:
        if col in df_xgb.columns:
            col_xgb = col
            break
    if col_xgb is None:
        print(f"[ERROR] No se hallo la columna de probabilidad en predictions.csv. Columnas: {list(df_xgb.columns)}")
        return

    # 3. NORMALIZACION Y LIMPIEZA DE LLAVES ESPACIOTEMPORALES
    print("\n[2/5] Estandarizando llaves geograficas y temporales...")
    for name, df in modelos_disponibles.items():
        df['fecha'] = pd.to_datetime(df['fecha']).dt.strftime('%Y-%m-%d')
        df['lat'] = df['lat'].round(2)
        df['lon'] = df['lon'].round(2)

    df_xgb_clean = df_xgb[['fecha', 'lat', 'lon', 'helada', col_xgb]].rename(
        columns={col_xgb: 'prob_xgb', 'helada': 'real_helada'}
    )
    df_rf_clean = df_rf[['fecha', 'lat', 'lon', 'prob_frost_rf']].rename(
        columns={'prob_frost_rf': 'prob_rf'}
    )

    # 4. UNIFICACION POR INTERSECCION (MERGE CONSECUTIVO)
    print("[3/5] Fusionando matrices mediante interseccion indexada...")
    final_df = pd.merge(df_xgb_clean, df_rf_clean, on=['fecha', 'lat', 'lon'], how='inner')

    if 'lstm' in modelos_disponibles:
        df_lstm_clean = modelos_disponibles['lstm'][['fecha', 'lat', 'lon', 'prob_helada_lstm']].rename(
            columns={'prob_helada_lstm': 'prob_lstm'}
        )
        final_df = pd.merge(final_df, df_lstm_clean, on=['fecha', 'lat', 'lon'], how='inner')

    if 'svm' in modelos_disponibles:
        df_svm_clean = modelos_disponibles['svm'][['fecha', 'lat', 'lon', 'prob_helada_svm']].rename(
            columns={'prob_helada_svm': 'prob_svm'}
        )
        final_df = pd.merge(final_df, df_svm_clean, on=['fecha', 'lat', 'lon'], how='inner')

    if final_df.empty:
        print("[ERROR] El cruce final quedo vacio. Revisa inconsistencias en fechas o coordenadas.")
        return

    # 5. ASIGNACION DINAMICA DE PESOS METODOLOGICOS
    print("[4/5] Calculando combinacion ponderada segun disponibilidad...")

    if 'lstm' in modelos_disponibles and 'svm' in modelos_disponibles:
        final_df['prob_helada_ensemble'] = (
            (final_df['prob_xgb'] * 0.30) +
            (final_df['prob_lstm'] * 0.30) +
            (final_df['prob_rf'] * 0.20) +
            (final_df['prob_svm'] * 0.20)
        )
    elif 'lstm' in modelos_disponibles:
        final_df['prob_helada_ensemble'] = (
            (final_df['prob_xgb'] * 0.35) +
            (final_df['prob_rf'] * 0.30) +
            (final_df['prob_lstm'] * 0.35)
        )
    elif 'svm' in modelos_disponibles:
        final_df['prob_helada_ensemble'] = (
            (final_df['prob_xgb'] * 0.40) +
            (final_df['prob_rf'] * 0.30) +
            (final_df['prob_svm'] * 0.30)
        )
    else:
        final_df['prob_helada_ensemble'] = (
            (final_df['prob_xgb'] * 0.55) +
            (final_df['prob_rf'] * 0.45)
        )

    final_df['pred_helada_ensemble'] = (final_df['prob_helada_ensemble'] >= 0.50).astype(int)

    # 6. EVALUACION Y DIAGNOSTICO ESTADISTICO
    print("[5/5] Extrayendo metricas de rendimiento del ensamble...")

    y_true = final_df['real_helada']
    y_pred = final_df['pred_helada_ensemble']

    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 70)
    print("RESULTADOS DEL MODELO DE ENSAMBLE METEOROLOGICO")
    print("=" * 70)
    print(f"F1-Score Combinado: {f1:.4f}")
    print(f"\nMatriz de Confusion Colectiva:\n{cm}")
    print(f"\nReporte de Rendimiento para el Articulo:\n{classification_report(y_true, y_pred, zero_division=0)}")

    # 7. EXPORTACION HOMOGENEA
    cols_a_guardar = ['fecha', 'lat', 'lon', 'real_helada', 'prob_xgb', 'prob_rf']
    if 'lstm' in modelos_disponibles:
        cols_a_guardar.append('prob_lstm')
    if 'svm' in modelos_disponibles:
        cols_a_guardar.append('prob_svm')
    cols_a_guardar.extend(['prob_helada_ensemble', 'pred_helada_ensemble'])

    output_df = final_df[cols_a_guardar]
    output_path = 'data_process/predictions_ensemble.csv'
    output_df.to_csv(output_path, index=False)

    print("=" * 70)
    print(f"[OK] Proceso de ensamble finalizado!")
    print(f"Registros regionales unificados: {len(output_df)}")
    print(f"Dataset guardado para generacion de mapas en: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    ensamblar_con_diagnostico()
