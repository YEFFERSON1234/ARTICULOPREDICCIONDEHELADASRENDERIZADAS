import pandas as pd
import numpy as np
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def ensamblar_con_diagnostico():
    print("Iniciando ensamble de modelos...")
    
    # 1. CARGAR DATOS
    try:
        df_xgb = pd.read_csv('data_process/predictions.csv')
        df_rf = pd.read_csv('data_process/predictions_rf.csv')
    except FileNotFoundError as e:
        print(f"[ERROR] No se encontraron archivos de predicciones: {e}")
        print("Ejecuta primero: python modelos/xgboost_model.py y python modelos/randomforest.py")
        return
    
    # Verificar si LSTM está disponible
    try:
        df_lstm = pd.read_csv('data_process/predictions_lstm.csv')
        has_lstm = True
        print("[INFO] LSTM disponible para ensamble")
    except FileNotFoundError:
        has_lstm = False
        print("[INFO] LSTM no disponible, ensamble solo XGBoost + RF")

    # 2. NORMALIZACIÓN DE FECHAS
    df_xgb['fecha'] = pd.to_datetime(df_xgb['fecha']).dt.date
    df_rf['fecha'] = pd.to_datetime(df_rf['fecha']).dt.date
    
    if has_lstm:
        df_lstm['fecha'] = pd.to_datetime(df_lstm['fecha']).dt.date

    # 3. SELECCIÓN DE COLUMNAS CLAVE
    df_xgb = df_xgb[['fecha', 'lat', 'lon', 'prob_helada']].rename(columns={'prob_helada': 'prob_xgb'})
    df_rf = df_rf[['fecha', 'lat', 'lon', 'prob_frost_rf']].rename(columns={'prob_frost_rf': 'prob_rf'})
    
    if has_lstm:
        df_lstm = df_lstm[['fecha', 'lat', 'lon', 'prob_helada_lstm']].rename(columns={'prob_helada_lstm': 'prob_lstm'})

    # 4. UNIFICACIÓN POR INTERSECCIÓN
    if has_lstm:
        m1 = pd.merge(df_xgb, df_rf, on=['fecha', 'lat', 'lon'], how='inner')
        final_df = pd.merge(m1, df_lstm, on=['fecha', 'lat', 'lon'], how='inner')
        # Pesos: XGBoost 35%, RF 30%, LSTM 35%
        final_df['prob_helada_final'] = (
            (final_df['prob_xgb'] * 0.35) + 
            (final_df['prob_rf'] * 0.30) + 
            (final_df['prob_lstm'] * 0.35)
        )
    else:
        # Solo XGBoost + RF
        final_df = pd.merge(df_xgb, df_rf, on=['fecha', 'lat', 'lon'], how='inner')
        # Pesos: XGBoost 55%, RF 45% (XGBoost tuvo mejor rendimiento)
        final_df['prob_helada_final'] = (
            (final_df['prob_xgb'] * 0.55) + 
            (final_df['prob_rf'] * 0.45)
        )

    if final_df.empty:
        print("\n[ERROR] No hay registros en común entre los archivos.")
        return

    # 5. GUARDAR RESULTADO FINAL
    final_df = final_df[['fecha', 'lat', 'lon', 'prob_helada_final']]
    final_df = final_df.rename(columns={'prob_helada_final': 'prob_helada'})
    final_df.to_csv('data_process/predictions_ensemble.csv', index=False)
    
    # También actualizar el predictions.csv principal
    final_df.to_csv('data_process/predictions.csv', index=False)
    
    print(f"\n[OK] Ensamble creado con {len(final_df)} registros")
    print(f"Archivo guardado en: data_process/predictions.csv")
    print(f"Modelos usados: XGBoost + RF" + (" + LSTM" if has_lstm else ""))

if __name__ == "__main__":
    ensamblar_con_diagnostico()