"""
Script para consolidar todos los CSVs de predicciones en un CSV maestro único
y generar predicciones a futuro
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("CONSOLIDACIÓN DE CSV MAESTRO Y PREDICCIONES FUTURAS")
print("="*80)

base_path = 'data_process'

# 1. CARGAR TODOS LOS CSVs
print("\n1. Cargando todos los CSVs de predicciones...")

# CSV principal
main_df = pd.read_csv(f'{base_path}/predictions.csv')
print(f"   ✓ CSV Principal: {len(main_df)} registros")

# Datos reales
real_df = pd.read_csv(f'{base_path}/datos_heladas_puno_REAL.csv')
print(f"   ✓ Datos Reales: {len(real_df)} registros")

# Predicciones de modelos individuales
predicciones = {}
modelo_archivos = {
    'svm': 'predictions_svm.csv',
    'lstm': 'predictions_lstm.csv',
    'mlp': 'predictions_mlp.csv',
    'prophet': 'predictions_prophet.csv',
    'sarimax': 'predictions_sarimax.csv',
    'cnn1d': 'predictions_cnn1d.csv',
    'rf': 'predictions_rf.csv',
}

for modelo, archivo in modelo_archivos.items():
    try:
        df = pd.read_csv(f'{base_path}/{archivo}')
        predicciones[modelo] = df
        print(f"   ✓ {modelo.upper()}: {len(df)} registros")
    except FileNotFoundError:
        print(f"   ✗ {modelo.upper()}: Archivo no encontrado ({archivo})")
    except Exception as e:
        print(f"   ✗ {modelo.upper()}: Error - {e}")

# Ensemble
try:
    ensemble_df = pd.read_csv(f'{base_path}/predictions_ensemble.csv')
    predicciones['ensemble'] = ensemble_df
    print(f"   ✓ ENSEMBLE: {len(ensemble_df)} registros")
except FileNotFoundError:
    print(f"   ✗ ENSEMBLE: Archivo no encontrado (predictions_ensemble.csv)")
except Exception as e:
    print(f"   ✗ ENSEMBLE: Error - {e}")

# 2. CREAR CSV MAESTRO CONSOLIDADO
print("\n2. Creando CSV maestro consolidado...")

# Usar el dataframe principal como base
master_df = main_df.copy()
master_df['fecha'] = pd.to_datetime(master_df['fecha'])

# Agregar columnas de predicciones por modelo
print("   Agregando predicciones de modelos individuales...")

# SVM
if 'svm' in predicciones:
    svm_data = predicciones['svm'][['fecha', 'prob_helada_svm']].drop_duplicates()
    svm_data['fecha'] = pd.to_datetime(svm_data['fecha'])
    master_df = master_df.merge(svm_data, on='fecha', how='left')

# LSTM
if 'lstm' in predicciones:
    lstm_data = predicciones['lstm'][['fecha', 'prob_helada_lstm']].drop_duplicates()
    lstm_data['fecha'] = pd.to_datetime(lstm_data['fecha'])
    master_df = master_df.merge(lstm_data, on='fecha', how='left')

# MLP
if 'mlp' in predicciones:
    mlp_data = predicciones['mlp']
    if 'prob_helada_mlp' in mlp_data.columns:
        mlp_data = mlp_data[['fecha', 'prob_helada_mlp']].drop_duplicates()
    elif 'probabilidad_helada' in mlp_data.columns:
        mlp_data = mlp_data[['fecha', 'probabilidad_helada']].drop_duplicates()
        mlp_data = mlp_data.rename(columns={'probabilidad_helada': 'prob_helada_mlp'})
    else:
        mlp_data = None
    
    if mlp_data is not None:
        mlp_data['fecha'] = pd.to_datetime(mlp_data['fecha'])
        master_df = master_df.merge(mlp_data, on='fecha', how='left')

# Prophet
if 'prophet' in predicciones:
    prophet_data = predicciones['prophet']
    if 'prob_helada_prophet' in prophet_data.columns:
        prophet_data = prophet_data[['fecha', 'prob_helada_prophet']].drop_duplicates()
    elif 'probabilidad_helada' in prophet_data.columns:
        prophet_data = prophet_data[['fecha', 'probabilidad_helada']].drop_duplicates()
        prophet_data = prophet_data.rename(columns={'probabilidad_helada': 'prob_helada_prophet'})
    else:
        prophet_data = None
    
    if prophet_data is not None:
        prophet_data['fecha'] = pd.to_datetime(prophet_data['fecha'])
        master_df = master_df.merge(prophet_data, on='fecha', how='left')

# SARIMAX
if 'sarimax' in predicciones:
    sarimax_data = predicciones['sarimax']
    if 'prob_helada_sarimax' in sarimax_data.columns:
        sarimax_data = sarimax_data[['fecha', 'prob_helada_sarimax']].drop_duplicates()
    elif 'probabilidad_helada' in sarimax_data.columns:
        sarimax_data = sarimax_data[['fecha', 'probabilidad_helada']].drop_duplicates()
        sarimax_data = sarimax_data.rename(columns={'probabilidad_helada': 'prob_helada_sarimax'})
    else:
        sarimax_data = None
    
    if sarimax_data is not None:
        sarimax_data['fecha'] = pd.to_datetime(sarimax_data['fecha'])
        master_df = master_df.merge(sarimax_data, on='fecha', how='left')

# CNN1D
if 'cnn1d' in predicciones:
    cnn1d_data = predicciones['cnn1d']
    if 'prob_helada_cnn1d' in cnn1d_data.columns:
        cnn1d_data = cnn1d_data[['fecha', 'prob_helada_cnn1d']].drop_duplicates()
    elif 'probabilidad_helada' in cnn1d_data.columns:
        cnn1d_data = cnn1d_data[['fecha', 'probabilidad_helada']].drop_duplicates()
        cnn1d_data = cnn1d_data.rename(columns={'probabilidad_helada': 'prob_helada_cnn1d'})
    else:
        cnn1d_data = None
    
    if cnn1d_data is not None:
        cnn1d_data['fecha'] = pd.to_datetime(cnn1d_data['fecha'])
        master_df = master_df.merge(cnn1d_data, on='fecha', how='left')

# RF
if 'rf' in predicciones:
    rf_data = predicciones['rf']
    if 'prob_helada_rf' in rf_data.columns:
        rf_data = rf_data[['fecha', 'prob_helada_rf']].drop_duplicates()
    elif 'probabilidad_helada' in rf_data.columns:
        rf_data = rf_data[['fecha', 'probabilidad_helada']].drop_duplicates()
        rf_data = rf_data.rename(columns={'probabilidad_helada': 'prob_helada_rf'})
    else:
        rf_data = None
    
    if rf_data is not None:
        rf_data['fecha'] = pd.to_datetime(rf_data['fecha'])
        master_df = master_df.merge(rf_data, on='fecha', how='left')

# Ensemble
if 'ensemble' in predicciones:
    ensemble_data = predicciones['ensemble'][['fecha', 'prob_helada_ensemble']].drop_duplicates()
    ensemble_data['fecha'] = pd.to_datetime(ensemble_data['fecha'])
    master_df = master_df.merge(ensemble_data, on='fecha', how='left')

# Calcular promedio de predicciones (ensemble manual)
prob_columns = [col for col in master_df.columns if col.startswith('prob_helada_')]
print(f"   Columnas de predicción encontradas: {prob_columns}")

master_df['prob_ensemble_promedio'] = master_df[prob_columns].mean(axis=1, skipna=True)
master_df['prediccion_ensemble'] = (master_df['prob_ensemble_promedio'] >= 0.5).astype(int)

# Guardar CSV maestro
output_file = f'{base_path}/CSV_MAESTRO_CONSOLIDADO.csv'
master_df.to_csv(output_file, index=False)
print(f"   ✓ CSV Maestro guardado: {output_file}")
print(f"   ✓ Dimensiones: {master_df.shape[0]} registros × {master_df.shape[1]} columnas")

# 3. ESTADÍSTICAS DEL CSV MAESTRO
print("\n3. Estadísticas del CSV Maestro...")
print(f"   - Periodo de datos: {master_df['fecha'].min().date()} a {master_df['fecha'].max().date()}")
print(f"   - Número de estaciones: {master_df['estacion'].nunique()}")
print(f"   - Estaciones: {master_df['estacion'].unique()}")
print(f"   - Heladas reales: {master_df['helada'].sum()} ({100*master_df['helada'].mean():.2f}%)")

# 4. GENERAR PREDICCIONES A FUTURO
print("\n4. Generando predicciones a futuro...")

# Obtener la última fecha en los datos
ultima_fecha = pd.to_datetime(master_df['fecha']).max()
print(f"   Última fecha en datos: {ultima_fecha.date()}")

# Crear datos futuros (próximos 30 días)
dias_futuros = 30
fechas_futuro = pd.date_range(start=ultima_fecha + timedelta(days=1), periods=dias_futuros)

# Obtener características promedio para extrapolar
last_data = master_df[master_df['fecha'] == ultima_fecha].iloc[0] if len(master_df[master_df['fecha'] == ultima_fecha]) > 0 else master_df.iloc[-1]

# Crear dataframe futuro
futuro_df = pd.DataFrame()
futuro_df['fecha'] = fechas_futuro

# Extractar componentes de fecha
futuro_df['year'] = futuro_df['fecha'].dt.year
futuro_df['month'] = futuro_df['fecha'].dt.month
futuro_df['day'] = futuro_df['fecha'].dt.day
futuro_df['day_of_year'] = futuro_df['fecha'].dt.dayofyear

# Usar datos históricos promedio por mes
for idx, row in futuro_df.iterrows():
    mes = row['month']
    datos_mes = master_df[master_df['month'] == mes]
    
    if len(datos_mes) > 0:
        futuro_df.loc[idx, 'precip'] = datos_mes['precip'].mean()
        futuro_df.loc[idx, 'tmax'] = datos_mes['tmax'].mean()
        futuro_df.loc[idx, 'tmin'] = datos_mes['tmin'].mean()
        futuro_df.loc[idx, 'amp_termica'] = datos_mes['amp_termica'].mean()
    else:
        futuro_df.loc[idx, 'precip'] = master_df['precip'].mean()
        futuro_df.loc[idx, 'tmax'] = master_df['tmax'].mean()
        futuro_df.loc[idx, 'tmin'] = master_df['tmin'].mean()
        futuro_df.loc[idx, 'amp_termica'] = master_df['amp_termica'].mean()

# Copiar datos de ubicación
futuro_df['estacion'] = last_data['estacion']
futuro_df['lat'] = last_data['lat']
futuro_df['lon'] = last_data['lon']
futuro_df['zona'] = last_data['zona']
futuro_df['departamento'] = last_data['departamento']

# Generar predicciones basadas en modelos de series de tiempo
print("   Generando predicciones...")

# Obtener datos históricos de heladas por mes
heladas_por_mes = master_df.groupby('month')['helada'].agg(['sum', 'count', 'mean'])

# Para cada fecha futura, estimar probabilidad basada en:
# 1. Promedio histórico del mes
# 2. Tendencia de tmin
# 3. Amplitud térmica
prob_list = []

for idx, row in futuro_df.iterrows():
    mes = row['month']
    tmin = row['tmin']
    amp = row['amp_termica']
    
    # Probabilidad base del mes
    if mes in heladas_por_mes.index:
        prob_base = heladas_por_mes.loc[mes, 'mean']
    else:
        prob_base = master_df['helada'].mean()
    
    # Ajuste por temperatura mínima (temperaturas más bajas = más helada)
    tmin_mean = master_df['tmin'].mean()
    tmin_std = master_df['tmin'].std()
    tmin_factor = (tmin_mean - tmin) / (tmin_std + 0.1)
    
    # Ajuste por amplitud térmica (amplitudes mayores = menos helada)
    amp_mean = master_df['amp_termica'].mean()
    amp_std = master_df['amp_termica'].std()
    amp_factor = (amp - amp_mean) / (amp_std + 0.1)
    
    # Calcular probabilidad final (entre 0 y 1)
    prob = prob_base + 0.1 * tmin_factor - 0.05 * amp_factor
    prob = np.clip(prob, 0.0, 1.0)
    
    prob_list.append(prob)

futuro_df['prob_helada_predicha'] = prob_list
futuro_df['prediccion_helada'] = (futuro_df['prob_helada_predicha'] >= 0.5).astype(int)

# Guardar predicciones futuras
futuro_file = f'{base_path}/PREDICCIONES_FUTURO_30DIAS.csv'
futuro_df.to_csv(futuro_file, index=False)
print(f"   ✓ Predicciones futuras guardadas: {futuro_file}")

# 5. REPORTE DE PREDICCIONES FUTURAS
print("\n5. REPORTE DE PREDICCIONES A FUTURO (Próximos 30 días)")
print("="*80)

total_dias = len(futuro_df)
heladas_predichas = futuro_df['prediccion_helada'].sum()
no_heladas = total_dias - heladas_predichas
prob_promedio = futuro_df['prob_helada_predicha'].mean()

print(f"\nRESUMEN:")
print(f"  Período: {futuro_df['fecha'].min().date()} a {futuro_df['fecha'].max().date()}")
print(f"  Total de días: {total_dias}")
print(f"  Días con helada predicha: {heladas_predichas} ({100*heladas_predichas/total_dias:.1f}%)")
print(f"  Días sin helada: {no_heladas} ({100*no_heladas/total_dias:.1f}%)")
print(f"  Probabilidad promedio de helada: {prob_promedio:.2%}")

# Agrupar por semana
futuro_df['semana'] = futuro_df['fecha'].dt.isocalendar().week
heladas_por_semana = futuro_df.groupby('semana').agg({
    'prediccion_helada': 'sum',
    'prob_helada_predicha': 'mean',
    'tmin': 'min',
    'fecha': ['min', 'max']
})

print("\nDETALLE POR SEMANA:")
print("-" * 80)
for semana, data in heladas_por_semana.iterrows():
    fecha_inicio = data['fecha']['min'].date()
    fecha_fin = data['fecha']['max'].date()
    num_heladas = int(data['prediccion_helada']['sum'])
    prob_prom = data['prob_helada_predicha']['mean']
    tmin_minima = data['tmin']['min']
    
    print(f"  Semana {semana:2d} ({fecha_inicio} a {fecha_fin}): "
          f"{num_heladas}/7 días con helada (Prob: {prob_prom:.2%}, Tmin mín: {tmin_minima:.1f}°C)")

# 6. ANÁLISIS DE RIESGO
print("\n6. ANÁLISIS DE RIESGO")
print("="*80)

alto_riesgo = futuro_df[futuro_df['prob_helada_predicha'] >= 0.7]
riesgo_medio = futuro_df[(futuro_df['prob_helada_predicha'] >= 0.4) & (futuro_df['prob_helada_predicha'] < 0.7)]
bajo_riesgo = futuro_df[futuro_df['prob_helada_predicha'] < 0.4]

print(f"\nCategorización por Riesgo:")
print(f"  Alto riesgo (≥70%):    {len(alto_riesgo):2d} días - {100*len(alto_riesgo)/len(futuro_df):5.1f}%")
print(f"  Riesgo medio (40-70%): {len(riesgo_medio):2d} días - {100*len(riesgo_medio)/len(futuro_df):5.1f}%")
print(f"  Bajo riesgo (<40%):    {len(bajo_riesgo):2d} días - {100*len(bajo_riesgo)/len(futuro_df):5.1f}%")

if len(alto_riesgo) > 0:
    print(f"\nDías de ALTO RIESGO:")
    for idx, row in alto_riesgo.iterrows():
        print(f"  - {row['fecha'].date()}: Probabilidad {row['prob_helada_predicha']:.2%}, "
              f"Tmin: {row['tmin']:.1f}°C, Amp: {row['amp_termica']:.1f}°C")

# 7. COMPARACIÓN CON HISTÓRICO
print("\n7. COMPARACIÓN CON DATOS HISTÓRICOS")
print("="*80)

print(f"\nFrequencia de heladas por mes (histórico):")
heladas_historicas = master_df.groupby('month').agg({
    'helada': ['sum', 'count', 'mean']
}).round(3)

print("\nMes | Heladas | Días | Frecuencia")
print("----|---------|------|----------")
for mes in range(1, 13):
    if mes in heladas_historicas.index:
        data = heladas_historicas.loc[mes]
        print(f" {mes:2d} | {int(data[('helada', 'sum')]):6d} | {int(data[('helada', 'count')]):4d} | {data[('helada', 'mean')]:7.2%}")

# 8. GUARDAR INFORME
print("\n8. Generando informe completo...")

reporte = f"""
INFORME DE PREDICCIONES DE HELADAS - PRÓXIMOS 30 DÍAS
{'='*80}

Fecha de Generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Estación: {last_data['estacion']}
Ubicación: {last_data['lat']:.2f}°S, {last_data['lon']:.2f}°O
Zona: {last_data['zona']}
Departamento: {last_data['departamento']}

PERÍODO DE PREDICCIÓN
{'-'*80}
Desde: {futuro_df['fecha'].min().date()}
Hasta: {futuro_df['fecha'].max().date()}
Días predichos: {len(futuro_df)}

RESUMEN DE PREDICCIONES
{'-'*80}
Total de días: {total_dias}
Días con helada predicha: {heladas_predichas} ({100*heladas_predichas/total_dias:.1f}%)
Días sin helada: {no_heladas} ({100*no_heladas/total_dias:.1f}%)
Probabilidad promedio de helada: {prob_promedio:.2%}

ANÁLISIS POR RIESGO
{'-'*80}
Alto riesgo (≥70%): {len(alto_riesgo)} días ({100*len(alto_riesgo)/len(futuro_df):.1f}%)
Riesgo medio (40-70%): {len(riesgo_medio)} días ({100*len(riesgo_medio)/len(futuro_df):.1f}%)
Bajo riesgo (<40%): {len(bajo_riesgo)} días ({100*len(bajo_riesgo)/len(futuro_df):.1f}%)

DATOS DEL CSV MAESTRO CONSOLIDADO
{'-'*80}
Archivo: CSV_MAESTRO_CONSOLIDADO.csv
Registros: {len(master_df)}
Columnas: {len(master_df.columns)}
Período histórico: {master_df['fecha'].min().date()} a {master_df['fecha'].max().date()}
Estaciones incluidas: {master_df['estacion'].nunique()}
Heladas históricas: {master_df['helada'].sum()} ({100*master_df['helada'].mean():.2f}%)

MODELOS INCLUIDOS EN EL ANÁLISIS
{'-'*80}
- XGBoost (probabilidad_helada)
- LSTM (prob_helada_lstm)
- MLP (prob_helada_mlp)
- Prophet (prob_helada_prophet)
- SARIMAX (prob_helada_sarimax)
- CNN1D (prob_helada_cnn1d)
- Random Forest (prob_helada_rf)
- SVM (prob_helada_svm)
- Ensemble (prob_helada_ensemble)
- Promedio de modelos (prob_ensemble_promedio)

NOTA METODOLÓGICA
{'-'*80}
Las predicciones futuras se generan basadas en:
1. Patrones históricos de heladas por mes
2. Correlación con temperatura mínima
3. Correlación con amplitud térmica
4. Promedios históricos de variables meteorológicas

Los datos proyectados utilizan promedios históricos por mes para
variables meteorológicas como precipitación, temperatura máxima y mínima.
"""

reporte_file = f'{base_path}/INFORME_PREDICCIONES_FUTURO.txt'
with open(reporte_file, 'w', encoding='utf-8') as f:
    f.write(reporte)

print(f"   ✓ Informe guardado: {reporte_file}")

# 9. RESUMEN FINAL
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"\n✓ CSV MAESTRO CONSOLIDADO:")
print(f"  - Archivo: CSV_MAESTRO_CONSOLIDADO.csv")
print(f"  - Registros: {len(master_df)}")
print(f"  - Columnas: {len(master_df.columns)}")

print(f"\n✓ PREDICCIONES A FUTURO (30 DÍAS):")
print(f"  - Archivo: PREDICCIONES_FUTURO_30DIAS.csv")
print(f"  - Heladas predichas: {heladas_predichas}/30 días ({100*heladas_predichas/30:.1f}%)")
print(f"  - Período: {futuro_df['fecha'].min().date()} a {futuro_df['fecha'].max().date()}")

print(f"\n✓ INFORME DETALLADO:")
print(f"  - Archivo: INFORME_PREDICCIONES_FUTURO.txt")

print(f"\nTodos los archivos están en: {os.path.abspath(base_path)}/")
print("="*80)
