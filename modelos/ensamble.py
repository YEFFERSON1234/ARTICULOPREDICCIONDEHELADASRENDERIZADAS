import pandas as pd
import numpy as np
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def ensamblar_con_diagnostico():
    print("="*70)
    print("INICIANDO ENSAMBLE DE MODELOS - DATASET REAL")
    print("="*70)
    
    # 1. CARGAR DATOS
    try:
        df_xgb = pd.read_csv('data_process/predictions.csv')
        df_rf = pd.read_csv('data_process/predictions_rf.csv')
        print("[OK] Archivos de XGBoost y Random Forest cargados con éxito.")
    except FileNotFoundError as e:
        print(f"[ERROR] No se encontraron archivos de predicciones: {e}")
        print("Asegúrate de que existan: data_process/predictions.csv y data_process/predictions_rf.csv")
        return
    
    # Verificar si LSTM está disponible
    try:
        df_lstm = pd.read_csv('data_process/predictions_lstm.csv')
        has_lstm = True
        print("[INFO] LSTM disponible para ensamble")
    except FileNotFoundError:
        has_lstm = False
        print("[INFO] LSTM no disponible, ensamble solo XGBoost + RF")

    # 2. SELECCIÓN DE NOMBRE DE COLUMNA PARA XGBOOST
    # Tu dataset usa 'probabilidad_helada', pero ponemos alternativas por si acaso
    col_xgb = None
    for col in ['probabilidad_helada', 'prob_helada', 'probabilidad']:
        if col in df_xgb.columns:
            col_xgb = col
            break
            
    if col_xgb is None:
        print(f"[ERROR] No se encontró la columna de probabilidad en predictions.csv.")
        print(f"Columnas detectadas en tu archivo: {list(df_xgb.columns)}")
        return
    else:
        print(f"[INFO] Detectada columna de probabilidad XGBoost: '{col_xgb}'")

    # 3. NORMALIZACIÓN DE FECHAS A STRING (Para evitar fallos de zona horaria en el merge)
    df_xgb['fecha'] = pd.to_datetime(df_xgb['fecha']).dt.strftime('%Y-%m-%d')
    df_rf['fecha'] = pd.to_datetime(df_rf['fecha']).dt.strftime('%Y-%m-%d')
    if has_lstm:
        df_lstm['fecha'] = pd.to_datetime(df_lstm['fecha']).dt.strftime('%Y-%m-%d')

    # 4. FILTRADO Y RENOMBRADO DE COLUMNAS CLAVE
    # Redondeamos lat/lon a 2 decimales para evitar desajustes numéricos flotantes (ej. -14.7900001 frente a -14.79)
    df_xgb['lat'] = df_xgb['lat'].round(2)
    df_xgb['lon'] = df_xgb['lon'].round(2)
    df_rf['lat'] = df_rf['lat'].round(2)
    df_rf['lon'] = df_rf['lon'].round(2)

    df_xgb_clean = df_xgb[['fecha', 'lat', 'lon', col_xgb]].rename(columns={col_xgb: 'prob_xgb'})
    df_rf_clean = df_rf[['fecha', 'lat', 'lon', 'prob_frost_rf']].rename(columns={'prob_frost_rf': 'prob_rf'})
    
    if has_lstm:
        df_lstm['lat'] = df_lstm['lat'].round(2)
        df_lstm['lon'] = df_lstm['lon'].round(2)
        df_lstm_clean = df_lstm[['fecha', 'lat', 'lon', 'prob_helada_lstm']].rename(columns={'prob_helada_lstm': 'prob_lstm'})

    # 5. UNIFICACIÓN POR INTERSECCIÓN (MERGE)
    print("[INFO] Combinando matrices de predicción por estación y fecha...")
    if has_lstm:
        m1 = pd.merge(df_xgb_clean, df_rf_clean, on=['fecha', 'lat', 'lon'], how='inner')
        final_df = pd.merge(m1, df_lstm_clean, on=['fecha', 'lat', 'lon'], how='inner')
        
        # Pesos equilibrados: XGBoost 35%, RF 30%, LSTM 35%
        final_df['prob_helada_final'] = (
            (final_df['prob_xgb'] * 0.35) + 
            (final_df['prob_rf'] * 0.30) + 
            (final_df['prob_lstm'] * 0.35)
        )
    else:
        # Solo XGBoost + RF
        final_df = pd.merge(df_xgb_clean, df_rf_clean, on=['fecha', 'lat', 'lon'], how='inner')
        
        # Pesos: XGBoost 55%, RF 45% (XGBoost suele tener mayor precisión en gradientes)
        final_df['prob_helada_final'] = (
            (final_df['prob_xgb'] * 0.55) + 
            (final_df['prob_rf'] * 0.45)
        )

    if final_df.empty:
        print("\n[ERROR] El ensamble quedó vacío.")
        print("Revisa si las coordenadas 'lat' y 'lon' o los formatos de 'fecha' difieren entre tus scripts.")
        return

    # 6. GUARDAR RESULTADOS EXCLUSIVOS
    # Conservamos columnas de localización y el veredicto final
    final_df = final_df[['fecha', 'lat', 'lon', 'prob_xgb', 'prob_rf', 'prob_helada_final']]
    final_df = final_df.rename(columns={'prob_helada_final': 'prob_helada_ensemble'})
    
    output_path = 'data_process/predictions_ensemble.csv'
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print(f"[OK] ¡Ensamble completado exitosamente!")
    print(f"Registros emparejados: {len(final_df)}")
    print(f"Resultados exportados a: {output_path}")
    print("="*70)

if __name__ == "__main__":
    ensamblar_con_diagnostico()