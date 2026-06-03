"""
Script para graficar predicciones de heladas para mañana
Genera visualizaciones de temperatura predicha y probabilidad de helada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
GRAFICOS_DIR = Path('graficos_resultados')
GRAFICOS_DIR.mkdir(exist_ok=True)

print("="*80)
print("GRAFICANDO PREDICCIÓN DE HELADAS PARA MAÑANA")
print("="*80)

# 1. CARGAR DATOS DE PREDICCIÓN
print("\n[1/4] Cargando datos de predicción...")
df = pd.read_csv('data_process/prediccion_manana.csv')
df['fecha'] = pd.to_datetime(df['fecha'])
fecha_prediccion = df['fecha'].iloc[0].strftime('%Y-%m-%d')
print(f"  Fecha de predicción: {fecha_prediccion}")
print(f"  Estaciones: {len(df)}")

# 2. GRÁFICO 1: TEMPERATURA PREDICHA POR ESTACIÓN
print("\n[2/4] Generando gráfico de temperatura predicha...")

fig, ax = plt.subplots(figsize=(14, 8))

# Ordenar por temperatura predicha
df_temp = df.sort_values('tmin_pred')

# Colores según temperatura (rojo para heladas, azul para temperaturas normales)
colors = ['red' if t <= 0 else 'orange' if t <= 2 else 'steelblue' for t in df_temp['tmin_pred']]

bars = ax.barh(df_temp['estacion'], df_temp['tmin_pred'], color=colors, alpha=0.7, edgecolor='black')

# Línea de congelación
ax.axvline(x=0, color='blue', linestyle='--', linewidth=2, label='Punto de congelación (0°C)')

ax.set_xlabel('Temperatura Mínima Predicha (°C)', fontsize=12, fontweight='bold')
ax.set_ylabel('Estación', fontsize=12, fontweight='bold')
ax.set_title(f'Temperatura Mínima Predicha por Estación - {fecha_prediccion}', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

# Añadir etiquetas de valor
for i, (idx, row) in enumerate(df_temp.iterrows()):
    ax.text(row['tmin_pred'] + 0.3, i, f"{row['tmin_pred']:.1f}°C", 
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'prediccion_manana_temperatura.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: graficos_resultados/prediccion_manana_temperatura.png")
plt.close()

# 3. GRÁFICO 2: PROBABILIDAD DE HELADA POR ESTACIÓN
print("\n[3/4] Generando gráfico de probabilidad de helada...")

fig, ax = plt.subplots(figsize=(14, 8))

# Ordenar por probabilidad de helada
df_prob = df.sort_values('prob_helada', ascending=True)

# Colores según nivel de riesgo
colors = ['darkred' if p >= 0.7 else 'orange' if p >= 0.3 else 'green' for p in df_prob['prob_helada']]

bars = ax.barh(df_prob['estacion'], df_prob['prob_helada'], color=colors, alpha=0.7, edgecolor='black')

# Línea de umbral (50%)
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Umbral de decisión (50%)')

ax.set_xlabel('Probabilidad de Helada', fontsize=12, fontweight='bold')
ax.set_ylabel('Estación', fontsize=12, fontweight='bold')
ax.set_title(f'Probabilidad de Helada por Estación - {fecha_prediccion}', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

# Añadir etiquetas de valor
for i, (idx, row) in enumerate(df_prob.iterrows()):
    ax.text(row['prob_helada'] + 0.02, i, f"{row['prob_helada']:.2f}", 
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'prediccion_manana_probabilidad.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: graficos_resultados/prediccion_manana_probabilidad.png")
plt.close()

# 4. GRÁFICO 3: MAPA DE RIESGO GEOGRÁFICO
print("\n[4/4] Generando mapa de riesgo geográfico...")

fig, ax = plt.subplots(figsize=(12, 10))

# Scatter plot con colores según probabilidad de helada
scatter = ax.scatter(df['lon'], df['lat'], 
                     c=df['prob_helada'], 
                     s=300, 
                     cmap='RdYlGn_r', 
                     alpha=0.7,
                     edgecolors='black',
                     linewidths=1.5,
                     vmin=0, vmax=1)

# Añadir etiquetas de estaciones
for idx, row in df.iterrows():
    ax.annotate(row['estacion'], 
                (row['lon'], row['lat']), 
                fontsize=8, 
                ha='center', 
                va='bottom',
                fontweight='bold')

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Probabilidad de Helada', fontsize=11, fontweight='bold')

ax.set_xlabel('Longitud', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitud', fontsize=12, fontweight='bold')
ax.set_title(f'Mapa de Riesgo de Heladas - {fecha_prediccion}', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Invertir eje Y (latitud negativa)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'prediccion_manana_mapa_riesgo.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: graficos_resultados/prediccion_manana_mapa_riesgo.png")
plt.close()

# 5. GRÁFICO 4: RESUMEN DE RIESGO
print("\n[5/5] Generando gráfico de resumen de riesgo...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f'Resumen de Riesgo de Heladas - {fecha_prediccion}', fontsize=16, fontweight='bold')

# Gráfico de pie - distribución de riesgo
riesgo_alto = len(df[df['prob_helada'] >= 0.7])
riesgo_medio = len(df[(df['prob_helada'] >= 0.3) & (df['prob_helada'] < 0.7)])
riesgo_bajo = len(df[df['prob_helada'] < 0.3])

sizes = [riesgo_alto, riesgo_medio, riesgo_bajo]
labels = ['Riesgo ALTO\n(≥70%)', 'Riesgo MEDIO\n(30-70%)', 'Riesgo BAJO\n(<30%)']
colors = ['darkred', 'orange', 'green']
explode = (0.1, 0, 0)

axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0].set_title('Distribución de Riesgo por Estación', fontweight='bold', fontsize=12)

# Gráfico de barras - estadísticas
temp_promedio = df['tmin_pred'].mean()
temp_min = df['tmin_pred'].min()
temp_max = df['tmin_pred'].max()
prob_promedio = df['prob_helada'].mean()

estadisticas = {
    'Temp. Promedio': temp_promedio,
    'Temp. Mínima': temp_min,
    'Temp. Máxima': temp_max,
    'Prob. Promedio': prob_promedio * 100
}

bars = axes[1].bar(estadisticas.keys(), estadisticas.values(), 
                   color=['steelblue', 'darkblue', 'lightblue', 'coral'], 
                   alpha=0.7, edgecolor='black')

axes[1].set_ylabel('Valor', fontsize=11, fontweight='bold')
axes[1].set_title('Estadísticas de Predicción', fontweight='bold', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

# Añadir etiquetas de valor
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'prediccion_manana_resumen.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Gráfico guardado: graficos_resultados/prediccion_manana_resumen.png")
plt.close()

# RESUMEN FINAL
print("\n" + "="*80)
print("RESUMEN DE GRÁFICOS GENERADOS")
print("="*80)
print(f"\n[OK] Todos los gráficos guardados en: graficos_resultados/")
print(f"\nGráficos generados:")
print(f"  1. prediccion_manana_temperatura.png - Temperatura predicha por estación")
print(f"  2. prediccion_manana_probabilidad.png - Probabilidad de helada por estación")
print(f"  3. prediccion_manana_mapa_riesgo.png - Mapa geográfico de riesgo")
print(f"  4. prediccion_manana_resumen.png - Resumen de riesgo y estadísticas")
print(f"\nEstadísticas de la predicción:")
print(f"  - Estaciones con riesgo ALTO: {riesgo_alto}")
print(f"  - Estaciones con riesgo MEDIO: {riesgo_medio}")
print(f"  - Estaciones con riesgo BAJO: {riesgo_bajo}")
print(f"  - Temperatura promedio predicha: {temp_promedio:.2f}°C")
print(f"  - Temperatura mínima predicha: {temp_min:.2f}°C")
print(f"  - Temperatura máxima predicha: {temp_max:.2f}°C")
print(f"  - Probabilidad promedio de helada: {prob_promedio*100:.1f}%")
print("\n" + "="*80)
print("¡GRAFICACIÓN COMPLETADA!")
print("="*80)
