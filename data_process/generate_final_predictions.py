import pandas as pd
import numpy as np
import sys

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("GENERANDO PREDICTIONS.CSV CON FORMATO CORRECTO")
print("="*70)

# Cargar predicciones de XGBoost
print("\n[1/3] Cargando predicciones XGBoost...")
xgb_preds = pd.read_csv('data_process/predictions.csv')
print(f"   Registros XGBoost: {len(xgb_preds)}")

# Cargar predicciones de Random Forest
print("\n[2/3] Cargando predicciones Random Forest...")
rf_preds = pd.read_csv('data_process/predictions_rf.csv')
print(f"   Registros RF: {len(rf_preds)}")

# Crear predictions.csv final con formato requerido: lat, lon, fecha, prob_helada
print("\n[3/3] Creando predictions.csv final...")

# Usar XGBoost como principal (mejor RMSE)
predictions_final = xgb_preds[['lat', 'lon', 'fecha', 'probabilidad_helada']].copy()
predictions_final = predictions_final.rename(columns={'probabilidad_helada': 'prob_helada'})

# Asegurar formato de fecha
predictions_final['fecha'] = pd.to_datetime(predictions_final['fecha']).dt.strftime('%Y-%m-%d')

# Guardar
predictions_final.to_csv('data_process/predictions.csv', index=False)

print(f"\n" + "="*70)
print(f"RESULTADOS")
print(f"="*70)
print(f"Total de registros: {len(predictions_final)}")
print(f"Columnas: {list(predictions_final.columns)}")
print(f"\nPrimeras filas:")
print(predictions_final.head())
print(f"\nEstadísticas de prob_helada:")
print(predictions_final['prob_helada'].describe())
print(f"\n[OK] predictions.csv guardado en: data_process/predictions.csv")
