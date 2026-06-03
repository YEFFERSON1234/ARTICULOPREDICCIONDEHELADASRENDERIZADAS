"""
Crear informe general consolidado con ejemplos específicos de mañana y pasado mañana
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("="*100)
print(" "*30 + "INFORME GENERAL DE PREDICCIONES - CONSOLIDADO")
print("="*100)

base_path = 'data_process'

# Cargar datos
maestro_df = pd.read_csv(f'{base_path}/CSV_MAESTRO_CONSOLIDADO.csv')
maestro_df['fecha'] = pd.to_datetime(maestro_df['fecha'])

futuro_df = pd.read_csv(f'{base_path}/PREDICCIONES_FUTURO_30DIAS.csv')
futuro_df['fecha'] = pd.to_datetime(futuro_df['fecha'])

# Definir fechas clave
ultima_fecha_historico = maestro_df['fecha'].max()
manana = ultima_fecha_historico + timedelta(days=1)
pasado_manana = ultima_fecha_historico + timedelta(days=2)

print(f"\n{'INFORMACIÓN GENERAL':-^100}")
print(f"\nFecha de Generación del Informe: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Última fecha de datos históricos: {ultima_fecha_historico.strftime('%Y-%m-%d')}")
print(f"\n▶ MAÑANA: {manana.strftime('%A, %d de %B de %Y')} ({manana.strftime('%Y-%m-%d')})")
print(f"▶ PASADO MAÑANA: {pasado_manana.strftime('%A, %d de %B de %Y')} ({pasado_manana.strftime('%Y-%m-%d')})")

# ============================================================================
# DATOS DEL CSV MAESTRO - ÚLTIMA SEMANA
# ============================================================================
print(f"\n{'='*100}")
print(f"{'ANÁLISIS HISTÓRICO - ÚLTIMA SEMANA DE DATOS':-^100}")
print(f"{'='*100}")

ultima_semana = maestro_df[maestro_df['fecha'] >= (ultima_fecha_historico - timedelta(days=7))].sort_values('fecha')

print(f"\nPeriodo: {ultima_semana['fecha'].min().strftime('%Y-%m-%d')} a {ultima_semana['fecha'].max().strftime('%Y-%m-%d')}")
print(f"Días en periodo: {len(ultima_semana) / 3} (3 estaciones × 7 días)\n")

print(f"{'FECHA':<12} {'ESTACION':<15} {'TMIN':<8} {'TMAX':<8} {'AMP':<8} {'HELADA':<8} {'PROB_XGB':<12} {'PROB_ENS':<12}")
print("-" * 100)

for idx, row in ultima_semana.iterrows():
    fecha_str = row['fecha'].strftime('%Y-%m-%d')
    estacion = row['estacion'][:13] if pd.notna(row['estacion']) else 'N/A'
    tmin = f"{row['tmin']:.1f}°C" if pd.notna(row['tmin']) else 'N/A'
    tmax = f"{row['tmax']:.1f}°C" if pd.notna(row['tmax']) else 'N/A'
    amp = f"{row['amp_termica']:.1f}°C" if pd.notna(row['amp_termica']) else 'N/A'
    helada = "SÍ" if row['helada'] == 1 else "NO"
    prob_xgb = f"{row['probabilidad_helada']:.2%}" if pd.notna(row['probabilidad_helada']) else 'N/A'
    prob_ens = f"{row['prob_ensemble_promedio']:.2%}" if pd.notna(row['prob_ensemble_promedio']) else 'N/A'
    
    print(f"{fecha_str:<12} {estacion:<15} {tmin:<8} {tmax:<8} {amp:<8} {helada:<8} {prob_xgb:<12} {prob_ens:<12}")

# ============================================================================
# PREDICCIONES PARA MAÑANA
# ============================================================================
print(f"\n{'='*100}")
print(f"{'📅 PREDICCIONES PARA MAÑANA - ' + manana.strftime('%d de %B de %Y'):-^100}")
print(f"{'='*100}")

manana_pred = futuro_df[futuro_df['fecha'] == manana]

if len(manana_pred) > 0:
    row_manana = manana_pred.iloc[0]
    
    print(f"\n🏢 ESTACIÓN: {row_manana['estacion']}")
    print(f"📍 UBICACIÓN: {row_manana['lat']:.2f}°S, {row_manana['lon']:.2f}°O")
    print(f"🗺️  ZONA: {row_manana['zona']} - {row_manana['departamento']}")
    
    print(f"\n📊 VARIABLES METEOROLÓGICAS PREDICHAS:")
    print(f"   • Temperatura Mínima: {row_manana['tmin']:.2f}°C")
    print(f"   • Temperatura Máxima: {row_manana['tmax']:.2f}°C")
    print(f"   • Amplitud Térmica: {row_manana['amp_termica']:.2f}°C")
    print(f"   • Precipitación: {row_manana['precip']:.2f} mm")
    
    print(f"\n🎯 PREDICCIÓN DE HELADA:")
    prob_pred = row_manana['prob_helada_predicha']
    if prob_pred >= 0.7:
        nivel_riesgo = "🔴 ALTO RIESGO"
    elif prob_pred >= 0.4:
        nivel_riesgo = "🟠 RIESGO MEDIO"
    else:
        nivel_riesgo = "🟢 BAJO RIESGO"
    
    print(f"   Probabilidad: {prob_pred:.2%}")
    print(f"   Nivel de Riesgo: {nivel_riesgo}")
    print(f"   Predicción Binaria: {'⚠️  HELADA ESPERADA' if row_manana['prediccion_helada'] == 1 else '✅ SIN HELADA'}")
    
    print(f"\n📈 COMPARACIÓN CON HISTÓRICO (Mismo mes/día):")
    # Buscar datos históricos similares
    mismo_mes_dia = maestro_df[(maestro_df['month'] == manana.month) & 
                                (maestro_df['day'] == manana.day)]
    
    if len(mismo_mes_dia) > 0:
        tmin_historico = mismo_mes_dia['tmin'].mean()
        heladas_historicas = mismo_mes_dia['helada'].mean()
        print(f"   • Tmin promedio histórico (mismo día): {tmin_historico:.2f}°C")
        print(f"   • Frecuencia de heladas (histórico): {heladas_historicas:.2%}")
        print(f"   • Diferencia de temperatura: {row_manana['tmin'] - tmin_historico:+.2f}°C")
    
    # Comparar con última fecha
    ultima_dia = maestro_df[maestro_df['fecha'] == ultima_fecha_historico].iloc[0]
    print(f"\n   Comparación con última fecha ({ultima_fecha_historico.strftime('%Y-%m-%d')}):")
    print(f"   • Tmin: {ultima_dia['tmin']:.2f}°C → {row_manana['tmin']:.2f}°C (cambio: {row_manana['tmin'] - ultima_dia['tmin']:+.2f}°C)")
    print(f"   • Amplitud térmica: {ultima_dia['amp_termica']:.2f}°C → {row_manana['amp_termica']:.2f}°C")

# ============================================================================
# PREDICCIONES PARA PASADO MAÑANA
# ============================================================================
print(f"\n{'='*100}")
print(f"{'📅 PREDICCIONES PARA PASADO MAÑANA - ' + pasado_manana.strftime('%d de %B de %Y'):-^100}")
print(f"{'='*100}")

pasado_manana_pred = futuro_df[futuro_df['fecha'] == pasado_manana]

if len(pasado_manana_pred) > 0:
    row_pasado_manana = pasado_manana_pred.iloc[0]
    
    print(f"\n🏢 ESTACIÓN: {row_pasado_manana['estacion']}")
    print(f"📍 UBICACIÓN: {row_pasado_manana['lat']:.2f}°S, {row_pasado_manana['lon']:.2f}°O")
    print(f"🗺️  ZONA: {row_pasado_manana['zona']} - {row_pasado_manana['departamento']}")
    
    print(f"\n📊 VARIABLES METEOROLÓGICAS PREDICHAS:")
    print(f"   • Temperatura Mínima: {row_pasado_manana['tmin']:.2f}°C")
    print(f"   • Temperatura Máxima: {row_pasado_manana['tmax']:.2f}°C")
    print(f"   • Amplitud Térmica: {row_pasado_manana['amp_termica']:.2f}°C")
    print(f"   • Precipitación: {row_pasado_manana['precip']:.2f} mm")
    
    print(f"\n🎯 PREDICCIÓN DE HELADA:")
    prob_pred_2 = row_pasado_manana['prob_helada_predicha']
    if prob_pred_2 >= 0.7:
        nivel_riesgo_2 = "🔴 ALTO RIESGO"
    elif prob_pred_2 >= 0.4:
        nivel_riesgo_2 = "🟠 RIESGO MEDIO"
    else:
        nivel_riesgo_2 = "🟢 BAJO RIESGO"
    
    print(f"   Probabilidad: {prob_pred_2:.2%}")
    print(f"   Nivel de Riesgo: {nivel_riesgo_2}")
    print(f"   Predicción Binaria: {'⚠️  HELADA ESPERADA' if row_pasado_manana['prediccion_helada'] == 1 else '✅ SIN HELADA'}")
    
    print(f"\n📈 COMPARACIÓN CON HISTÓRICO (Mismo mes/día):")
    mismo_mes_dia_2 = maestro_df[(maestro_df['month'] == pasado_manana.month) & 
                                  (maestro_df['day'] == pasado_manana.day)]
    
    if len(mismo_mes_dia_2) > 0:
        tmin_historico_2 = mismo_mes_dia_2['tmin'].mean()
        heladas_historicas_2 = mismo_mes_dia_2['helada'].mean()
        print(f"   • Tmin promedio histórico (mismo día): {tmin_historico_2:.2f}°C")
        print(f"   • Frecuencia de heladas (histórico): {heladas_historicas_2:.2%}")
        print(f"   • Diferencia de temperatura: {row_pasado_manana['tmin'] - tmin_historico_2:+.2f}°C")
    
    print(f"\n   Comparación con mañana:")
    print(f"   • Tmin: {row_manana['tmin']:.2f}°C → {row_pasado_manana['tmin']:.2f}°C (cambio: {row_pasado_manana['tmin'] - row_manana['tmin']:+.2f}°C)")
    print(f"   • Probabilidad de helada: {prob_pred:.2%} → {prob_pred_2:.2%}")

# ============================================================================
# TABLA COMPARATIVA: MAÑANA VS PASADO MAÑANA
# ============================================================================
print(f"\n{'='*100}")
print(f"{'TABLA COMPARATIVA: MAÑANA vs PASADO MAÑANA':-^100}")
print(f"{'='*100}\n")

print(f"{'VARIABLE':<25} {'MAÑANA':<30} {'PASADO MAÑANA':<30}")
print("-" * 85)

if len(manana_pred) > 0 and len(pasado_manana_pred) > 0:
    row_m = manana_pred.iloc[0]
    row_pm = pasado_manana_pred.iloc[0]
    
    print(f"{'Fecha':<25} {manana.strftime('%Y-%m-%d'):<30} {pasado_manana.strftime('%Y-%m-%d'):<30}")
    print(f"{'Temperatura Mínima':<25} {row_m['tmin']:.2f}°C{'':<23} {row_pm['tmin']:.2f}°C")
    print(f"{'Temperatura Máxima':<25} {row_m['tmax']:.2f}°C{'':<23} {row_pm['tmax']:.2f}°C")
    print(f"{'Amplitud Térmica':<25} {row_m['amp_termica']:.2f}°C{'':<23} {row_pm['amp_termica']:.2f}°C")
    print(f"{'Precipitación':<25} {row_m['precip']:.2f} mm{'':<23} {row_pm['precip']:.2f} mm")
    print(f"{'Prob. Helada':<25} {row_m['prob_helada_predicha']:.2%}{'':<24} {row_pm['prob_helada_predicha']:.2%}")
    print(f"{'Riesgo':<25} {'ALTO' if row_m['prob_helada_predicha'] >= 0.7 else 'MEDIO' if row_m['prob_helada_predicha'] >= 0.4 else 'BAJO':<30} {'ALTO' if row_pm['prob_helada_predicha'] >= 0.7 else 'MEDIO' if row_pm['prob_helada_predicha'] >= 0.4 else 'BAJO':<30}")
    print(f"{'Predicción':<25} {'SÍ - HELADA' if row_m['prediccion_helada'] == 1 else 'NO':<30} {'SÍ - HELADA' if row_pm['prediccion_helada'] == 1 else 'NO':<30}")

# ============================================================================
# ESTADÍSTICAS DEL CSV MAESTRO
# ============================================================================
print(f"\n{'='*100}")
print(f"{'📊 ESTADÍSTICAS DEL CSV MAESTRO CONSOLIDADO':-^100}")
print(f"{'='*100}")

print(f"\n📁 CONTENIDO DEL CSV MAESTRO:")
print(f"   • Total de registros: {len(maestro_df):,}")
print(f"   • Período cubierto: {maestro_df['fecha'].min().strftime('%Y-%m-%d')} a {maestro_df['fecha'].max().strftime('%Y-%m-%d')}")
print(f"   • Duración: {(maestro_df['fecha'].max() - maestro_df['fecha'].min()).days} días")
print(f"   • Número de estaciones: {maestro_df['estacion'].nunique()}")
print(f"   • Estaciones: {', '.join(maestro_df['estacion'].unique())}")

print(f"\n❄️  ESTADÍSTICAS DE HELADAS:")
total_heladas = maestro_df['helada'].sum()
total_registros = len(maestro_df)
pct_heladas = (total_heladas / total_registros) * 100

print(f"   • Total de heladas registradas: {int(total_heladas):,}")
print(f"   • Total sin heladas: {int(total_registros - total_heladas):,}")
print(f"   • Frecuencia de heladas: {pct_heladas:.2f}%")

print(f"\n🌡️  ESTADÍSTICAS DE TEMPERATURA MÍNIMA:")
print(f"   • Promedio: {maestro_df['tmin'].mean():.2f}°C")
print(f"   • Mínima: {maestro_df['tmin'].min():.2f}°C")
print(f"   • Máxima: {maestro_df['tmin'].max():.2f}°C")
print(f"   • Desv. Estándar: {maestro_df['tmin'].std():.2f}°C")

print(f"\n📈 DISTRIBUCIÓN DE HELADAS POR MES:")
heladas_por_mes = maestro_df.groupby('month').agg({
    'helada': ['sum', 'count', 'mean']
}).round(4)

meses = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DIC']
print(f"\n{'MES':<6} {'HELADAS':<10} {'DÍAS':<10} {'FRECUENCIA':<15}")
print("-" * 41)

for mes_num in range(1, 13):
    if mes_num in heladas_por_mes.index:
        data = heladas_por_mes.loc[mes_num]
        heladas = int(data[('helada', 'sum')])
        dias = int(data[('helada', 'count')])
        freq = data[('helada', 'mean')] * 100
        print(f"{meses[mes_num-1]:<6} {heladas:<10} {dias:<10} {freq:>6.2f}%")

# ============================================================================
# MODELOS EVALUADOS
# ============================================================================
print(f"\n{'='*100}")
print(f"{'🤖 MODELOS DE PREDICCIÓN INCLUIDOS EN CSV MAESTRO':-^100}")
print(f"{'='*100}")

modelos_info = {
    'XGBoost': 'probabilidad_helada',
    'LSTM': 'prob_helada_lstm',
    'MLP': 'prob_helada_mlp',
    'Prophet': 'prob_helada_prophet',
    'SARIMAX': 'prob_helada_sarimax',
    'CNN1D': 'prob_helada_cnn1d',
    'Random Forest': 'prob_helada_rf',
    'SVM': 'prob_helada_svm',
    'Ensemble': 'prob_helada_ensemble',
}

print("\nModelos presentes en el CSV maestro:\n")
for idx, (modelo, columna) in enumerate(modelos_info.items(), 1):
    if columna in maestro_df.columns:
        media = maestro_df[columna].mean()
        desv = maestro_df[columna].std()
        print(f"{idx}. {modelo:20s} - Media: {media:.4f}, Desv: {desv:.4f}")
    else:
        print(f"{idx}. {modelo:20s} - (no disponible)")

# Promedio de ensemble
if 'prob_ensemble_promedio' in maestro_df.columns:
    media_ens = maestro_df['prob_ensemble_promedio'].mean()
    desv_ens = maestro_df['prob_ensemble_promedio'].std()
    print(f"\n✨ Promedio de todos los modelos - Media: {media_ens:.4f}, Desv: {desv_ens:.4f}")

# ============================================================================
# RECOMENDACIONES Y ANÁLISIS
# ============================================================================
print(f"\n{'='*100}")
print(f"{'💡 RECOMENDACIONES Y ANÁLISIS':-^100}")
print(f"{'='*100}")

print(f"\n🔍 ANÁLISIS DE RIESGO PARA LOS PRÓXIMOS 2 DÍAS:\n")

if len(manana_pred) > 0 and len(pasado_manana_pred) > 0:
    row_m = manana_pred.iloc[0]
    row_pm = pasado_manana_pred.iloc[0]
    
    print(f"MAÑANA ({manana.strftime('%Y-%m-%d')}):")
    if row_m['prob_helada_predicha'] >= 0.7:
        print(f"   ⚠️  RIESGO ALTO - Implementar medidas de protección")
        print(f"   ✓ Cubrir plantas sensibles")
        print(f"   ✓ Riego preventivo si es necesario")
        print(f"   ✓ Monitoreo constante")
    elif row_m['prob_helada_predicha'] >= 0.4:
        print(f"   ⚠️  RIESGO MEDIO - Estar alerta")
        print(f"   ✓ Preparar equipos de protección")
        print(f"   ✓ Monitorear temperatura durante la noche")
        print(f"   ✓ Revisar sistemas de riego")
    else:
        print(f"   ✅ RIESGO BAJO - Operaciones normales")
        print(f"   ✓ Sin medidas especiales requeridas")
    
    print(f"\nPASADO MAÑANA ({pasado_manana.strftime('%Y-%m-%d')}):")
    if row_pm['prob_helada_predicha'] >= 0.7:
        print(f"   ⚠️  RIESGO ALTO - Implementar medidas de protección")
        print(f"   ✓ Cubrir plantas sensibles")
        print(f"   ✓ Riego preventivo si es necesario")
        print(f"   ✓ Monitoreo constante")
    elif row_pm['prob_helada_predicha'] >= 0.4:
        print(f"   ⚠️  RIESGO MEDIO - Estar alerta")
        print(f"   ✓ Preparar equipos de protección")
        print(f"   ✓ Monitorear temperatura durante la noche")
        print(f"   ✓ Revisar sistemas de riego")
    else:
        print(f"   ✅ RIESGO BAJO - Operaciones normales")
        print(f"   ✓ Sin medidas especiales requeridas")

# Tendencia
print(f"\n📊 TENDENCIA (Mañana → Pasado Mañana):")
if len(manana_pred) > 0 and len(pasado_manana_pred) > 0:
    cambio_tmin = row_pm['tmin'] - row_m['tmin']
    cambio_prob = row_pm['prob_helada_predicha'] - row_m['prob_helada_predicha']
    
    print(f"   Temperatura mínima: {cambio_tmin:+.2f}°C {'(Calentamiento)' if cambio_tmin > 0 else '(Enfriamiento)'}")
    print(f"   Riesgo de helada: {cambio_prob:+.2%} {'(Disminuye riesgo)' if cambio_prob < 0 else '(Aumenta riesgo)'}")
    
    if cambio_prob < -0.1:
        print(f"   ✅ Tendencia POSITIVA: El riesgo disminuirá")
    elif cambio_prob > 0.1:
        print(f"   ⚠️  Tendencia NEGATIVA: El riesgo aumentará")
    else:
        print(f"   → Tendencia ESTABLE: El riesgo se mantiene")

# ============================================================================
# INFORMACIÓN TÉCNICA
# ============================================================================
print(f"\n{'='*100}")
print(f"{'🔧 INFORMACIÓN TÉCNICA':-^100}")
print(f"{'='*100}")

print(f"\n📄 ARCHIVOS UTILIZADOS:")
print(f"   • CSV_MAESTRO_CONSOLIDADO.csv ({len(maestro_df):,} registros)")
print(f"   • PREDICCIONES_FUTURO_30DIAS.csv ({len(futuro_df):,} registros)")

print(f"\n📊 COLUMNAS DEL CSV MAESTRO ({len(maestro_df.columns)} columnas):")
for i, col in enumerate(maestro_df.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n🎯 METODOLOGÍA:")
print(f"   • Las predicciones futuras se basan en patrones históricos mensuales")
print(f"   • Se correlaciona con temperatura mínima y amplitud térmica")
print(f"   • Los datos meteorológicos usan promedios históricos por mes")
print(f"   • Validación mediante 5-fold cross-validation")
print(f"   • Modelo principal: XGBoost (AUC-ROC: 0.9620)")

print(f"\n{'='*100}")
print(f"{'FIN DEL INFORME':-^100}")
print(f"{'='*100}\n")

# Guardar informe en archivo
informe_file = f'{base_path}/INFORME_GENERAL_CONSOLIDADO.txt'
with open(informe_file, 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write(" "*30 + "INFORME GENERAL DE PREDICCIONES - CONSOLIDADO\n")
    f.write("="*100 + "\n")
    f.write(f"\nFecha de Generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Última fecha histórica: {ultima_fecha_historico.strftime('%Y-%m-%d')}\n")
    f.write(f"\nMañana: {manana.strftime('%Y-%m-%d')}\n")
    f.write(f"Pasado Mañana: {pasado_manana.strftime('%Y-%m-%d')}\n")
    f.write("\nSe generó análisis consolidado del CSV Maestro y Predicciones Futuras.\n")
    f.write("Ver salida de consola para detalles completos.\n")

print(f"✅ Informe general guardado en: {informe_file}")
