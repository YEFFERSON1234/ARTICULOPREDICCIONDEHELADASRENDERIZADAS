import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# =====================================================================
# 1. CONFIGURACIÓN DE RUTAS RELATIVAS
# =====================================================================
dir_actual = os.path.dirname(os.path.abspath(__file__))
raiz_proyecto = os.path.abspath(os.path.join(dir_actual, "..", "..")) if "modelos" in dir_actual else dir_actual

carpeta_predicciones = os.path.join(raiz_proyecto, 'Predicciones')
carpeta_mapas = os.path.join(raiz_proyecto, 'modelos', 'Mapas_Renderizados')
os.makedirs(carpeta_mapas, exist_ok=True)

archivos_modelos = {
    'XGBoost': 'predictions_xgb.csv',
    'Random Forest': 'predictions_rf.csv',
    'SVM': 'predictions_svm.csv'
}

# Límites geográficos aproximados de la región de Puno para encuadrar el mapa
LAT_MIN, LAT_MAX = -17.5, -13.0
LON_MIN, LON_MAX = -71.2, -68.0

# =====================================================================
# 2. BUCLE DE PROCESAMIENTO Y RENDERIZADO ESPACIAL
# =====================================================================
print("-> [1/2] Iniciando el proceso de interpolación espacial sobre Puno...")

for nombre, archivo in archivos_modelos.items():
    ruta_csv = os.path.join(carpeta_predicciones, archivo)
    
    if not os.path.exists(ruta_csv):
        print(f"[!] Advertencia: No se encontró {archivo} en /Predicciones. Saltando modelo.")
        continue
        
    print(f"   -> Renderizando matriz geográfica para: {nombre}")
    df = pd.read_csv(ruta_csv)
    
    # Extraer las coordenadas y la variable objetivo (Probabilidad de Helada)
    # Agrupamos por coordenada para obtener el riesgo promedio del periodo evaluado
    df_resumen = df.groupby(['Lat', 'Long'])['prob_helada'].mean().reset_index()
    
    x = df_resumen['Long'].values
    y = df_resumen['Lat'].values
    z = df_resumen['prob_helada'].values
    
    # Crear una malla regular (Grid) densa sobre Puno para suavizar los contornos del mapa
    grid_x, grid_y = np.mgrid[LON_MIN:LON_MAX:200j, LAT_MIN:LAT_MAX:200j]
    
    # Interpolación cúbica para generar superficies continuas y elegantes (Estilo Renderizado)
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    
    # =====================================================================
    # 3. DISEÑO DEL MAPA CIENTÍFICO (MATPLOTLIB)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(6.5, 7.5), dpi=300)
    
    # Graficar contornos llenos (Superficie de calor térmica)
    # Usamos la paleta 'Frost' (Ceilán/Azules a Rojos) o 'Blues_r' para denotar frío intenso
    niveles = np.linspace(0, 1, 21)
    mapa_calor = ax.contourf(grid_x, grid_y, grid_z, levels=niveles, cmap='coolwarm', extend='both')
    
    # Añadir líneas de contorno sutiles para resaltar gradientes de altitud/presión térmica
    contornos_lineas = ax.contour(grid_x, grid_y, grid_z, levels=[0.25, 0.5, 0.75], 
                                  colors='black', linewidths=0.5, alpha=0.5, linestyles='--')
    ax.clabel(contornos_lineas, inline=True, fontsize=8, fmt='%.2f')
    
    # Configuración de ejes con formato de coordenadas geográficas
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    
    ax.set_xlabel('Longitud (°O)', fontsize=10, weight='bold', labelpad=10)
    ax.set_ylabel('Latitud (°S)', fontsize=10, weight='bold', labelpad=10)
    
    # Referencia del Altiplano: Marcamos la ubicación aproximada del Lago Titicaca como control geográfico
    ax.text(-69.5, -15.8, 'Lago\nTiticaca', fontsize=9, color='#2c3e50', 
            weight='bold', style='italic', ha='center', bbox=dict(facecolor='white', alpha=0.4, boxstyle='round,pad=0.3'))
    
    # Barra de color predictiva (Colorbar) estilizada
    cbar = fig.colorbar(mapa_calor, ax=ax, orientation='horizontal', pad=0.1, aspect=30)
    cbar.set_label('Probabilidad Estimada de Helada ($Temp \leq 0^\circ C$)', fontsize=10, weight='bold', labelpad=8)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Detalles estéticos formales (Estilo revista científica)
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    ax.set_title(f'MAPA ESTIMATIVO ESPACIAL DE HELADAS\nModelo Predictivo: {nombre} (Grilla ERA5)', 
                 fontsize=11, weight='bold', pad=15, color='#2c3e50')
    
    # Guardar mapa final optimizado para impresión
    nombre_salida = f"mapa_renderizado_{nombre.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(carpeta_mapas, nombre_salida), bbox_inches='tight', dpi=300)
    plt.close()

# =====================================================================
# 4. CONTROL DE SALIDA
# =====================================================================
print("-> [2/2] ¡Proceso de renderizado completado con éxito!")
print(f"Encontrarás tus mapas geográficos guardados en formato PNG de alta resolución aquí:\n --> {carpeta_mapas}\n")