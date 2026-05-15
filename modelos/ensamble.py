import pandas as pd
import numpy as np

def ensamblar_con_diagnostico():
    print("Iniciando armonización de archivos CSV...")
    
    # 1. CARGAR DATOS
    df_xgb = pd.read_csv('limpiezadedatos/predictions.csv')      # tmin_pred
    df_rf = pd.read_csv('limpiezadedatos/predictions_rf.csv')   # tmin_pred_rf
    df_lstm = pd.read_csv('limpiezadedatos/predictions_lstm.csv') # tmin_pred_lstm

    # 2. NORMALIZACIÓN DE FECHAS Y ESTACIONES
    # Esto es vital para que el merge funcione
    for name, df in zip(['XGB', 'RF', 'LSTM'], [df_xgb, df_rf, df_lstm]):
        df['fecha'] = pd.to_datetime(df['fecha']).dt.date
        df['estacion'] = df['estacion'].str.upper().str.strip()
        print(f"-> {name}: {len(df)} registros desde {df['fecha'].min()} hasta {df['fecha'].max()}")

    # 3. SELECCIÓN DE COLUMNAS CLAVE
    # Limpiamos los archivos para quedarnos solo con lo necesario
    df_xgb = df_xgb[['fecha', 'estacion', 'lat', 'lon', 'tmin_pred']].rename(columns={'tmin_pred': 'pred_xgb'})
    df_rf = df_rf[['fecha', 'estacion', 'tmin_pred_rf']]
    df_lstm = df_lstm[['fecha', 'estacion', 'tmin_pred_lstm']]

    # 4. UNIFICACIÓN POR INTERSECCIÓN (Inner Join)
    # Buscamos solo los días que aparecen en los TRES archivos
    m1 = pd.merge(df_xgb, df_rf, on=['fecha', 'estacion'], how='inner')
    final_df = pd.merge(m1, df_lstm, on=['fecha', 'estacion'], how='inner')

    if final_df.empty:
        print("\n⚠️ ALERTA: No hay fechas en común entre los archivos.")
        print("Asegúrate de que todos los modelos usen el mismo año de prueba (ej. 2015).")
        return

    # 5. CÁLCULO DEL ENSAMBLE PONDERADO 
    # Damos pesos según la precisión que vimos anteriormente
    final_df['tmin_final'] = (
        (final_df['pred_xgb'] * 0.30) + 
        (final_df['tmin_pred_rf'] * 0.30) + 
        (final_df['tmin_pred_lstm'] * 0.40)
    )

    # 6. NIVEL DE RIESGO PARA OPENGL [cite: 12]
    # Clasificamos para el Heatmap: 2 (Crítico), 1 (Moderado), 0 (Bajo)
    final_df['prob_helada'] = np.where(final_df['tmin_final'] <= -5, 2, 
                                       np.where(final_df['tmin_final'] <= 0, 1, 0))

    # 7. GUARDAR RESULTADO FINAL
    final_df.to_csv('limpiezadedatos/final_frost_atlas.csv', index=False)
    print(f"\n✅ ¡ÉXITO! Ensamble creado con {len(final_df)} registros comunes.")
    print(f"Archivo listo para OpenGL: limpiezadedatos/final_frost_atlas.csv")

if __name__ == "__main__":
    ensamblar_con_diagnostico()