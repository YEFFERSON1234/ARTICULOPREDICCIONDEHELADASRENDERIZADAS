"""
Generar visualizaciones adicionales para las predicciones futuras
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

base_path = 'data_process'
output_dir = 'graficos_resultados'

print("="*80)
print("GENERANDO VISUALIZACIONES DE PREDICCIONES FUTURAS")
print("="*80)

# Cargar datos
print("\n1. Cargando datos...")
futuro_df = pd.read_csv(f'{base_path}/PREDICCIONES_FUTURO_30DIAS.csv')
maestro_df = pd.read_csv(f'{base_path}/CSV_MAESTRO_CONSOLIDADO.csv')
maestro_df['fecha'] = pd.to_datetime(maestro_df['fecha'])
futuro_df['fecha'] = pd.to_datetime(futuro_df['fecha'])

print(f"   ✓ Predicciones futuras: {len(futuro_df)} registros")
print(f"   ✓ CSV Maestro: {len(maestro_df)} registros")

# ============================================================================
# GRÁFICO 1: LÍNEA DE TIEMPO - PREDICCIONES FUTURAS
# ============================================================================
print("\n2. Generando gráfico de línea temporal...")
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Temperatura mínima
axes[0].plot(futuro_df['fecha'], futuro_df['tmin'], marker='o', linewidth=2, 
             markersize=6, color='steelblue', label='Tmin predicha')
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Punto de congelación')
axes[0].fill_between(futuro_df['fecha'], futuro_df['tmin'], 0, 
                      where=(futuro_df['tmin'] <= 0), alpha=0.3, color='red')
axes[0].set_ylabel('Temperatura (°C)', fontsize=11, fontweight='bold')
axes[0].set_title('Temperatura Mínima Predicha - Próximos 30 Días', fontsize=12, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

# Probabilidad de helada
colors = ['red' if p >= 0.7 else 'orange' if p >= 0.4 else 'green' for p in futuro_df['prob_helada_predicha']]
axes[1].bar(futuro_df['fecha'], futuro_df['prob_helada_predicha'], color=colors, alpha=0.7, edgecolor='black')
axes[1].axhline(y=0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Umbral (50%)')
axes[1].axhline(y=0.7, color='r', linestyle=':', linewidth=1.5, alpha=0.5, label='Alto riesgo (70%)')
axes[1].axhline(y=0.4, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='Riesgo medio (40%)')
axes[1].set_ylabel('Probabilidad', fontsize=11, fontweight='bold')
axes[1].set_title('Probabilidad de Helada Predicha', fontsize=12, fontweight='bold')
axes[1].set_ylim([0, 1.0])
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3, axis='y')

# Amplitud térmica
axes[2].plot(futuro_df['fecha'], futuro_df['amp_termica'], marker='s', linewidth=2, 
             markersize=6, color='coral', label='Amplitud térmica')
axes[2].fill_between(futuro_df['fecha'], futuro_df['amp_termica'], alpha=0.3, color='coral')
axes[2].set_ylabel('Amplitud (°C)', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Fecha', fontsize=11, fontweight='bold')
axes[2].set_title('Amplitud Térmica Predicha', fontsize=12, fontweight='bold')
axes[2].legend(loc='upper right')
axes[2].grid(alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/11_predicciones_futuro_linea_temporal.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 11_predicciones_futuro_linea_temporal.png")
plt.close()

# ============================================================================
# GRÁFICO 2: CALENDARIO DE HELADAS
# ============================================================================
print("\n3. Generando calendario de heladas...")
fig, ax = plt.subplots(figsize=(14, 8))

# Crear matriz para el calendario
dias = futuro_df['day'].values
fechas = pd.to_datetime(futuro_df['fecha']).values
probs = futuro_df['prob_helada_predicha'].values
predicciones = futuro_df['prediccion_helada'].values

# Crear tabla visual
semanas = []
semana_actual = []
for i, (fecha, prob, pred) in enumerate(zip(fechas, probs, predicciones)):
    # Convertir numpy.datetime64 a pandas Timestamp
    fecha_ts = pd.Timestamp(fecha)
    semana_actual.append({'fecha': fecha_ts, 'prob': prob, 'pred': pred, 'dia': dias[i]})
    
    if len(semana_actual) == 7 or i == len(fechas) - 1:
        semanas.append(semana_actual)
        semana_actual = []

# Visualizar como tabla de calendario
days_of_week = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

fig, axes = plt.subplots(len(semanas), 7, figsize=(16, 10))
fig.suptitle('Calendario de Predicción de Heladas (Próximos 30 Días)', 
             fontsize=14, fontweight='bold', y=0.98)

for week_idx, semana in enumerate(semanas):
    for day_idx in range(7):
        ax = axes[week_idx, day_idx] if len(semanas) > 1 else axes[day_idx]
        
        if day_idx < len(semana):
            data = semana[day_idx]
            prob = data['prob']
            pred = data['pred']
            fecha = data['fecha']
            
            # Color basado en probabilidad
            if prob >= 0.7:
                color = '#FF4444'  # Rojo - Alto riesgo
            elif prob >= 0.4:
                color = '#FFA500'  # Naranja - Riesgo medio
            else:
                color = '#44AA44'  # Verde - Bajo riesgo
            
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, edgecolor='black', linewidth=2))
            ax.text(0.5, 0.7, f"{fecha.strftime('%d')}", ha='center', va='center', 
                   fontsize=12, fontweight='bold')
            ax.text(0.5, 0.4, f"{prob:.0%}", ha='center', va='center', 
                   fontsize=10)
            ax.text(0.5, 0.15, "HELADA" if pred == 1 else "No helada", ha='center', va='center', 
                   fontsize=8, style='italic')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        
        # Añadir nombre del día en la primera semana
        if week_idx == 0:
            ax.text(0.5, 1.15, days_of_week[day_idx], ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/12_calendario_heladas_futuro.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 12_calendario_heladas_futuro.png")
plt.close()

# ============================================================================
# GRÁFICO 3: COMPARACIÓN HISTÓRICO VS FUTURO
# ============================================================================
print("\n4. Generando comparación histórico vs futuro...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Últimos 30 días históricos
ultimos_30 = maestro_df[maestro_df['fecha'].dt.strftime('%Y-%m') == '2015-10'].tail(30).copy()
ultimos_30 = ultimos_30.sort_values('fecha')

# Comparar temperatura mínima
axes[0, 0].plot(ultimos_30['fecha'], ultimos_30['tmin'], marker='o', linewidth=2, 
                label='Histórico', color='blue', alpha=0.7, markersize=5)
axes[0, 0].plot(futuro_df['fecha'], futuro_df['tmin'], marker='s', linewidth=2, 
                label='Futuro (Predicho)', color='red', alpha=0.7, markersize=5)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[0, 0].set_ylabel('Temperatura Mínima (°C)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Comparación: Tmin Histórico vs Futuro', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Frecuencia de heladas
hist_heladas = ultimos_30['helada'].sum()
fut_heladas = futuro_df['prediccion_helada'].sum()

categories = ['Últimos 30 días\n(Histórico)', 'Próximos 30 días\n(Predicción)']
values = [hist_heladas, fut_heladas]
colors_bar = ['steelblue', 'coral']

axes[0, 1].bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 1].set_ylabel('Número de Días con Helada', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Días con Helada: Histórico vs Futuro', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim([0, 31])
for i, v in enumerate(values):
    axes[0, 1].text(i, v + 0.5, f'{int(v)} días', ha='center', fontsize=11, fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='y')

# Amplitud térmica
axes[1, 0].plot(ultimos_30['fecha'], ultimos_30['amp_termica'], marker='o', linewidth=2, 
                label='Histórico', color='green', alpha=0.7, markersize=5)
axes[1, 0].plot(futuro_df['fecha'], futuro_df['amp_termica'], marker='s', linewidth=2, 
                label='Futuro (Predicho)', color='orange', alpha=0.7, markersize=5)
axes[1, 0].set_ylabel('Amplitud Térmica (°C)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Comparación: Amplitud Térmica', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Estadísticas
hist_tmin_mean = ultimos_30['tmin'].mean()
fut_tmin_mean = futuro_df['tmin'].mean()
hist_tmin_min = ultimos_30['tmin'].min()
fut_tmin_min = futuro_df['tmin'].min()

stats_text = f"""ESTADÍSTICAS COMPARATIVAS

Histórico (Últimos 30 días):
  • Tmin promedio: {hist_tmin_mean:.2f}°C
  • Tmin mínima: {hist_tmin_min:.2f}°C
  • Días con helada: {hist_heladas}
  • Frecuencia: {100*hist_heladas/len(ultimos_30):.1f}%

Futuro (Próximos 30 días):
  • Tmin promedio: {fut_tmin_mean:.2f}°C
  • Tmin mínima: {fut_tmin_min:.2f}°C
  • Días con helada: {fut_heladas}
  • Frecuencia: {100*fut_heladas/len(futuro_df):.1f}%

Diferencias:
  • ΔTmin promedio: {fut_tmin_mean - hist_tmin_mean:+.2f}°C
  • ΔDías helada: {fut_heladas - hist_heladas:+.0f} días
"""

axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/13_comparacion_historico_futuro.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 13_comparacion_historico_futuro.png")
plt.close()

# ============================================================================
# GRÁFICO 4: ANÁLISIS DE RIESGO CON HEATMAP
# ============================================================================
print("\n5. Generando heatmap de riesgo...")

# Crear matriz de semanas x días
semanas_data = []
for i in range(0, len(futuro_df), 7):
    semana = futuro_df.iloc[i:i+7]['prob_helada_predicha'].values
    if len(semana) < 7:
        semana = np.pad(semana, (0, 7-len(semana)), constant_values=np.nan)
    semanas_data.append(semana)

semanas_array = np.array(semanas_data)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(semanas_array, annot=True, fmt='.0%', cmap='RdYlGn_r', 
            xticklabels=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sab', 'Dom'],
            yticklabels=[f'Semana {i+1}' for i in range(len(semanas_array))],
            cbar_kws={'label': 'Probabilidad de Helada'}, vmin=0, vmax=1,
            linewidths=1, linecolor='gray', ax=ax)
ax.set_title('Heatmap de Probabilidad de Helada (Próximos 30 Días)', 
            fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('Semana', fontsize=12, fontweight='bold')
ax.set_xlabel('Día de la Semana', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/14_heatmap_riesgo_heladas.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 14_heatmap_riesgo_heladas.png")
plt.close()

# ============================================================================
# GRÁFICO 5: DISTRIBUCIÓN DE RIESGOS
# ============================================================================
print("\n6. Generando distribución de riesgos...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
alto_riesgo = len(futuro_df[futuro_df['prob_helada_predicha'] >= 0.7])
riesgo_medio = len(futuro_df[(futuro_df['prob_helada_predicha'] >= 0.4) & 
                              (futuro_df['prob_helada_predicha'] < 0.7)])
bajo_riesgo = len(futuro_df[futuro_df['prob_helada_predicha'] < 0.4])

sizes = [alto_riesgo, riesgo_medio, bajo_riesgo]
labels = [f'Alto (≥70%)\n{alto_riesgo} días', f'Medio (40-70%)\n{riesgo_medio} días', 
          f'Bajo (<40%)\n{bajo_riesgo} días']
colors_pie = ['#FF4444', '#FFA500', '#44AA44']

axes[0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0].set_title('Distribución de Riesgos', fontsize=12, fontweight='bold')

# Distribución de probabilidades
axes[1].hist(futuro_df['prob_helada_predicha'], bins=15, color='steelblue', 
            alpha=0.7, edgecolor='black')
axes[1].axvline(x=0.4, color='orange', linestyle='--', linewidth=2, label='Riesgo Medio')
axes[1].axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Alto Riesgo')
axes[1].set_xlabel('Probabilidad de Helada', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frecuencia (días)', fontsize=11, fontweight='bold')
axes[1].set_title('Histograma de Probabilidades', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/15_distribucion_riesgos.png', dpi=300, bbox_inches='tight')
print("   ✓ Guardado: 15_distribucion_riesgos.png")
plt.close()

print("\n" + "="*80)
print("RESUMEN DE VISUALIZACIONES GENERADAS")
print("="*80)
print(f"\n✓ Total de gráficos nuevos: 5")
print(f"  - 11_predicciones_futuro_linea_temporal.png")
print(f"  - 12_calendario_heladas_futuro.png")
print(f"  - 13_comparacion_historico_futuro.png")
print(f"  - 14_heatmap_riesgo_heladas.png")
print(f"  - 15_distribucion_riesgos.png")

print(f"\nArchivos de datos generados:")
print(f"  - CSV_MAESTRO_CONSOLIDADO.csv ({len(maestro_df)} registros)")
print(f"  - PREDICCIONES_FUTURO_30DIAS.csv ({len(futuro_df)} registros)")
print(f"  - INFORME_PREDICCIONES_FUTURO.txt")

print("\n" + "="*80)
