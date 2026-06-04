import pandas as pd
import numpy as np
import os
import folium
from folium.plugins import HeatMapWithTime

# =====================================================================
# 1. CONFIGURACIÓN DE RUTAS RELATIVAS
# =====================================================================
dir_actual = os.path.dirname(os.path.abspath(__file__))
raiz_proyecto = os.path.abspath(os.path.join(dir_actual, "..", "..")) if "modelos" in dir_actual else dir_actual

carpeta_predicciones = os.path.join(raiz_proyecto, 'Predicciones')
carpeta_mapas = os.path.join(raiz_proyecto, 'modelos', 'Mapas_Renderizados')
os.makedirs(carpeta_mapas, exist_ok=True)

ruta_salida_html = os.path.join(carpeta_mapas, 'mapa_interactivo_anual_assembler.html')
archivo_assembler = os.path.join(carpeta_predicciones, 'predictions_assembler.csv')

if not os.path.exists(archivo_assembler):
    print(f"[!] Error crítico: No se encontró 'predictions_assembler.csv' en {carpeta_predicciones}.")
    print("Asegúrate de ejecutar primero el script del Assembler para consolidar los pesos.")
    exit()

# =====================================================================
# 2. CARGA Y PROCESAMIENTO MACRO POR AÑOS
# =====================================================================
print("-> [1/3] Cargando predicciones del Assembler y extrayendo componentes anuales...")
df = pd.read_csv(archivo_assembler)

# Convertir a datetime y extraer el año como cadena para el índice del mapa
df['fecha'] = pd.to_datetime(df['fecha'])
df['anio'] = df['fecha'].dt.year

# Obtener la lista ordenada de años únicos disponibles en tu base de datos histórica
lista_anios = sorted(df['anio'].unique())
lista_anios_str = [str(a) for a in lista_anios]

print(f"   -> Años detectados en la serie temporal: {lista_anios_str}")

# Estructurar la matriz de calor indexada por año
datos_por_anio = []

for anio in lista_anios:
    # Filtrar datos del año actual
    df_anio = df[df['anio'] == anio]
    
    # Agrupamos por coordenadas para obtener el riesgo promedio consolidado de ese año específico
    df_mapa_anual = df_anio.groupby(['Lat', 'Long'])['prob_helada'].mean().reset_index()
    
    # Formato requerido por HeatMapWithTime: [[Lat, Long, Peso], [Lat, Long, Peso], ...]
    puntos_anio = df_mapa_anual[['Lat', 'Long', 'prob_helada']].values.tolist()
    datos_por_anio.append(puntos_anio)

# =====================================================================
# 3. CONSTRUCCIÓN DEL LIENZO INTERACTIVO SOBRE PUNO
# =====================================================================
print("-> [2/3] Configurando mapa base de OpenStreetMap centrado en el Altiplano...")

coordenadas_puno = [-15.20, -69.80]

mapa_base = folium.Map(
    location=coordenadas_puno,
    zoom_start=8,                  # Encuadre ideal para visualizar todo el departamento
    tiles='OpenStreetMap',
    control_scale=True
)

# =====================================================================
# 4. IMPLEMENTACIÓN DE LA CAPA HEATMAP CLIMÁTICA INTERANUAL
# =====================================================================
print("-> [3/3] Inyectando el motor dinámico de mapas de calor interanuales...")

capa_calor_anual = HeatMapWithTime(
    data=datos_por_anio,
    index=lista_anios_str,         # Los años aparecerán en el control deslizante
    radius=22,                     # Un radio ligeramente mayor para suavizar la grilla a nivel macro
    min_opacity=0.15,
    max_opacity=0.85,
    scale_radius=False,            # Mantiene la escala simétrica sin importar el zoom
    auto_play=False,               # Detenido por defecto para que el usuario explore manualmente
    display_index=True,            # Muestra el año actual en una etiqueta flotante grande
    index_steps=1,
    position='bottomleft'          # Control de reproducción abajo a la izquierda
)

# Acoplar la capa de tiempo al mapa interactivo y guardar
capa_calor_anual.add_to(mapa_base)
mapa_base.save(ruta_salida_html)

print("\n" + "="*70)
print("¡COMPLETADO CON ÉXITO! Se ha generado la animación interanual.")
print(f"Ruta del mapa web:\n --> {ruta_salida_html}")
print("="*70)
print("💡 MODO DE USO: Abre el archivo HTML en tu navegador y arrastra el control")
print("   deslizante inferior para observar la evolución climática año por año.")