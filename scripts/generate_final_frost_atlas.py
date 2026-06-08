import pandas as pd
import numpy as np

# Cargar el CSV maestro consolidado
print("Cargando CSV maestro consolidado...")
df_maestro = pd.read_csv('data_process/CSV_MAESTRO_CONSOLIDADO.csv')

# Crear el atlas de heladas con las columnas necesarias
print("Generando atlas de heladas final...")

# Seleccionar columnas relevantes
columnas_atlas = [
    'year', 'month', 'day', 'precip', 'tmax', 'tmin', 'estacion', 'lat', 'lon', 
    'zona', 'departamento', 'fecha', 'frost', 'day_of_year', 
    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3'
]

# Verificar si las columnas existen
columnas_disponibles = [col for col in columnas_atlas if col in df_maestro.columns]
df_atlas = df_maestro[columnas_disponibles].copy()

# Agregar predicciones del ensemble
if 'prob_ensemble_promedio' in df_maestro.columns:
    df_atlas['probabilidad_helada'] = df_maestro['prob_ensemble_promedio']
else:
    # Si no existe, usar la primera columna de probabilidad disponible
    prob_cols = [col for col in df_maestro.columns if 'prob' in col.lower() and 'helada' in col.lower()]
    if prob_cols:
        df_atlas['probabilidad_helada'] = df_maestro[prob_cols[0]]
    else:
        df_atlas['probabilidad_helada'] = 0.5

# Agregar predicción de temperatura
if 'tmin_pred' in df_maestro.columns:
    df_atlas['tmin_pred'] = df_maestro['tmin_pred']
else:
    df_atlas['tmin_pred'] = df_atlas['tmin']

# Calcular nivel de riesgo basado en probabilidad
def calcular_nivel_riesgo(prob):
    if pd.isna(prob):
        return 'Desconocido'
    elif prob < 0.2:
        return 'Muy Bajo'
    elif prob < 0.4:
        return 'Bajo'
    elif prob < 0.6:
        return 'Moderado'
    elif prob < 0.8:
        return 'Alto'
    else:
        return 'Muy Alto'

df_atlas['nivel_riesgo'] = df_atlas['probabilidad_helada'].apply(calcular_nivel_riesgo)

# Agregar predicciones individuales de modelos si están disponibles
if 'prob_helada_xgb' in df_maestro.columns:
    df_atlas['pred_xgb'] = df_maestro.get('prob_helada_xgb', 0)
else:
    df_atlas['pred_xgb'] = df_atlas['probabilidad_helada']

if 'tmin_pred_rf' in df_maestro.columns:
    df_atlas['tmin_pred_rf'] = df_maestro['tmin_pred_rf']
else:
    df_atlas['tmin_pred_rf'] = df_atlas['tmin_pred']

# Calcular temperatura final (promedio de predicciones disponibles)
temp_preds = [col for col in ['tmin_pred', 'tmin_pred_rf', 'tmin_pred_lstm', 'tmin_pred_xgb'] if col in df_atlas.columns]
if len(temp_preds) > 1:
    df_atlas['tmin_final'] = df_atlas[temp_preds].mean(axis=1)
else:
    df_atlas['tmin_final'] = df_atlas['tmin_pred']

# Eliminar duplicados manteniendo el primer registro
df_atlas = df_atlas.drop_duplicates(subset=['fecha', 'estacion'], keep='first')

# Guardar el atlas final
print("Guardando atlas de heladas final...")
output_path = 'data_process/final_frost_atlas.csv'
df_atlas.to_csv(output_path, index=False)

print(f"✅ Atlas de heladas final generado exitosamente: {output_path}")
print(f"📊 Total de registros: {len(df_atlas)}")
print(f"📍 Estaciones incluidas: {df_atlas['estacion'].nunique()}")
print(f"📅 Rango de fechas: {df_atlas['fecha'].min()} a {df_atlas['fecha'].max()}")
print(f"🌡️ Distribución de niveles de riesgo:")
print(df_atlas['nivel_riesgo'].value_counts())