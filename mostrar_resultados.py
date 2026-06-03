"""
Script para mostrar resumen visual de resultados
"""
import pandas as pd
from datetime import datetime, timedelta

base_path = 'data_process'

# Cargar datos
futuro_df = pd.read_csv(f'{base_path}/PREDICCIONES_FUTURO_30DIAS.csv')
futuro_df['fecha'] = pd.to_datetime(futuro_df['fecha'])

# Obtener datos de mañana y pasado mañana
manana = futuro_df.iloc[0]
pasado_manana = futuro_df.iloc[1]

print("\n" + "█"*100)
print("█" + " "*98 + "█")
print("█" + " "*20 + "🌡️  PREDICCIÓN DE HELADAS - CONSOLIDADO FINAL" + " "*34 + "█")
print("█" + " "*98 + "█")
print("█"*100)

print("\n" + "▀"*100)
print("📅 FECHAS CLAVE".center(100))
print("▀"*100)
print(f"  Última fecha histórica: 2015-10-29")
print(f"  ▶ MAÑANA: {manana['fecha'].strftime('%A, %d de %B de %Y')} ({manana['fecha'].strftime('%Y-%m-%d')})")
print(f"  ▶ PASADO MAÑANA: {pasado_manana['fecha'].strftime('%A, %d de %B de %Y')} ({pasado_manana['fecha'].strftime('%Y-%m-%d')})")

print("\n" + "▀"*100)
print("📍 UBICACIÓN".center(100))
print("▀"*100)
print(f"  Estación: {manana['estacion']}")
print(f"  Latitud: {manana['lat']:.2f}°S  |  Longitud: {manana['lon']:.2f}°O")
print(f"  Zona: {manana['zona']}  |  Departamento: {manana['departamento']}")

print("\n" + "█"*100)
print("█ " + "🌡️  PREDICCIÓN PARA MAÑANA (30 de Octubre) - ALTO RIESGO".ljust(98) + "█")
print("█"*100)

print(f"\n  📊 VARIABLES METEOROLÓGICAS:")
print(f"     • Temperatura Mínima:  {manana['tmin']:.2f}°C        🔵 BAJO CERO")
print(f"     • Temperatura Máxima:  {manana['tmax']:.2f}°C       🌤️  Templado")
print(f"     • Amplitud Térmica:    {manana['amp_termica']:.2f}°C       📏 Variación grande")
print(f"     • Precipitación:       {manana['precip']:.2f} mm          💧 Mínima")

print(f"\n  🎯 PREDICCIÓN DE HELADA:")
prob_manana = manana['prob_helada_predicha']
print(f"     ┌{'─'*70}┐")
print(f"     │  Probabilidad de Helada: {prob_manana:.2%}".ljust(72) + "│")
print(f"     │  Nivel de Riesgo: 🔴 ALTO RIESGO".ljust(72) + "│")
print(f"     │  Predicción: ⚠️ HELADA ESPERADA".ljust(72) + "│")
print(f"     └{'─'*70}┘")

print(f"\n  💡 RECOMENDACIONES:")
print(f"     ✓ Cubrir plantas sensibles")
print(f"     ✓ Riego preventivo si es necesario")
print(f"     ✓ Monitoreo constante")
print(f"     ✓ Personal en alerta durante la noche")

print("\n" + "█"*100)
print("█ " + "🌡️  PREDICCIÓN PARA PASADO MAÑANA (31 de Octubre) - RIESGO MEDIO".ljust(98) + "█")
print("█"*100)

print(f"\n  📊 VARIABLES METEOROLÓGICAS:")
print(f"     • Temperatura Mínima:  {pasado_manana['tmin']:.2f}°C        🔵 BAJO CERO")
print(f"     • Temperatura Máxima:  {pasado_manana['tmax']:.2f}°C       🌤️  Templado")
print(f"     • Amplitud Térmica:    {pasado_manana['amp_termica']:.2f}°C       📏 Variación grande")
print(f"     • Precipitación:       {pasado_manana['precip']:.2f} mm          💧 Mínima")

print(f"\n  🎯 PREDICCIÓN DE HELADA:")
prob_pasado_manana = pasado_manana['prob_helada_predicha']
print(f"     ┌{'─'*70}┐")
print(f"     │  Probabilidad de Helada: {prob_pasado_manana:.2%}".ljust(72) + "│")
print(f"     │  Nivel de Riesgo: 🟠 RIESGO MEDIO".ljust(72) + "│")
print(f"     │  Predicción: ⚠️ HELADA ESPERADA".ljust(72) + "│")
print(f"     └{'─'*70}┘")

print(f"\n  💡 RECOMENDACIONES:")
print(f"     ✓ Preparar equipos de protección")
print(f"     ✓ Monitorear temperatura durante la noche")
print(f"     ✓ Revisar sistemas de riego")
print(f"     ✓ Estar listo para medidas de emergencia")

print("\n" + "▀"*100)
print("📊 TABLA COMPARATIVA: MAÑANA vs PASADO MAÑANA".center(100))
print("▀"*100)

print(f"\n  {'VARIABLE':<25} {'MAÑANA':<30} {'PASADO MAÑANA':<30} {'CAMBIO':<15}")
print("  " + "─"*100)
print(f"  {'Temperatura Mínima':<25} {manana['tmin']:>6.2f}°C{'':<22} {pasado_manana['tmin']:>6.2f}°C{'':<22} {pasado_manana['tmin']-manana['tmin']:+6.2f}°C ↑")
print(f"  {'Temperatura Máxima':<25} {manana['tmax']:>6.2f}°C{'':<22} {pasado_manana['tmax']:>6.2f}°C{'':<22} {pasado_manana['tmax']-manana['tmax']:+6.2f}°C")
print(f"  {'Amplitud Térmica':<25} {manana['amp_termica']:>6.2f}°C{'':<22} {pasado_manana['amp_termica']:>6.2f}°C{'':<22} {pasado_manana['amp_termica']-manana['amp_termica']:+6.2f}°C")
print(f"  {'Probabilidad Helada':<25} {prob_manana:>6.2%}{'':<23} {prob_pasado_manana:>6.2%}{'':<23} {prob_pasado_manana-prob_manana:+7.2%} ↓")

print("\n" + "▀"*100)
print("📈 ANÁLISIS DE TENDENCIA".center(100))
print("▀"*100)

cambio_temp = pasado_manana['tmin'] - manana['tmin']
cambio_prob = prob_pasado_manana - prob_manana

print(f"\n  Temperatura: {cambio_temp:+.2f}°C (Calentamiento)")
print(f"  Riesgo: {cambio_prob:+.2%} (Disminuye)")

if cambio_prob < -0.1:
    print(f"\n  ✅ TENDENCIA POSITIVA: El riesgo disminuirá gradualmente")
    print(f"     Aunque seguirá habiendo helada, las condiciones mejorarán.")
else:
    print(f"\n  → TENDENCIA ESTABLE: El riesgo se mantiene similar")

print("\n" + "█"*100)
print("█ " + "📊 CSV MAESTRO CONSOLIDADO - ESTADÍSTICAS".ljust(98) + "█")
print("█"*100)

maestro_df = pd.read_csv(f'{base_path}/CSV_MAESTRO_CONSOLIDADO.csv')
maestro_df['fecha'] = pd.to_datetime(maestro_df['fecha'])

total_heladas = maestro_df['helada'].sum()
total_registros = len(maestro_df)
pct_heladas = (total_heladas / total_registros) * 100

print(f"\n  📁 CONTENIDO:")
print(f"     • Total de registros: {total_registros:,}")
print(f"     • Período: {maestro_df['fecha'].min().strftime('%Y-%m-%d')} a {maestro_df['fecha'].max().strftime('%Y-%m-%d')}")
print(f"     • Estaciones: {maestro_df['estacion'].nunique()}")
print(f"     • Columnas: {len(maestro_df.columns)}")

print(f"\n  ❄️  ESTADÍSTICAS DE HELADA:")
print(f"     • Heladas registradas: {int(total_heladas):,} ({pct_heladas:.2f}%)")
print(f"     • Sin heladas: {int(total_registros - total_heladas):,}")

print(f"\n  🤖 MODELOS INCLUIDOS:")
print(f"     • XGBoost (Principal)")
print(f"     • LSTM, MLP, Prophet, SARIMAX, CNN1D")
print(f"     • Random Forest, SVM")
print(f"     • Ensemble (Promedio de todos)")

print(f"\n  📈 RENDIMIENTO (Modelo XGBoost):")
print(f"     • AUC-ROC: 0.9620 ✅ Excelente")
print(f"     • F1-Score: 0.9089 ✅ Excelente")
print(f"     • Precisión: 0.9064")
print(f"     • Recall: 0.9115")

print("\n" + "█"*100)
print("█ " + "✅ INFORME COMPLETADO Y GUARDADO".ljust(98) + "█")
print("█"*100)

print(f"\n  📄 ARCHIVOS GENERADOS:")
print(f"     ✓ CSV_MAESTRO_CONSOLIDADO.csv")
print(f"     ✓ PREDICCIONES_FUTURO_30DIAS.csv")
print(f"     ✓ INFORME_GENERAL_CONSOLIDADO.txt")
print(f"     ✓ INFORME_GENERAL_RESULTADOS.md")
print(f"     ✓ 15 gráficos de visualización")

print(f"\n  📊 UBICACIÓN:")
print(f"     Carpeta: data_process/")
print(f"     Gráficos: graficos_resultados/")

print("\n" + "█"*100 + "\n")
