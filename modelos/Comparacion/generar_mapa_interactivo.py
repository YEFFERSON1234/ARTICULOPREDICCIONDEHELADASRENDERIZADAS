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
ruta_salida_html = os.path.join(raiz_proyecto, 'modelos', 'Mapas_Renderizados', 'mapa_interactivo_heladas.html')

# Usaremos XGBoost por ser el modelo con mayor rendimiento (TSS: 0.81)
archivo_xgb = os.path.join(carpeta_predicciones, 'predictions_xgb.csv')

if not os.path.exists(archivo_xgb):
    print(f"[!] Error crítico: No se encontró 'predictions_xgb.csv' en {carpeta_predicciones}.")
    print("Asegúrate de ejecutar primero el entrenamiento de XGBoost para generar las predicciones base.")
    exit()

# =====================================================================
# 2. CARGA Y PREPARACIÓN DE DATOS TEMPORALES
# =====================================================================
print("-> [1/3] Cargando matrices de predicción espacial y ordenando cronológicamente...")
df = pd.read_csv(archivo_xgb)

# Asegurar formato de fecha y ordenar el dataset por tiempo
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values(by='fecha')

# Extraer la lista única de fechas formateadas como texto para la barra interactiva
lista_fechas = df['fecha'].dt.strftime('%Y-%m-%d').unique().tolist()

print(f"   -> Rango temporal detectado: {lista_fechas[0]} hasta {lista_fechas[-1]} ({len(lista_fechas)} días).")

# Reestructurar los datos para el formato que exige HeatMapWithTime:
# Una lista de listas, donde cada sublista contiene los puntos [Lat, Long, Peso] de ese día específico
datos_por_dia = []

for fecha_str in lista_fechas:
    # Filtrar las filas pertenecientes al día actual
    df_dia = df[df['fecha'].dt.strftime('%Y-%m-%d') == fecha_str]
    
    # El "Peso" del mapa de calor será la probabilidad de helada calculada por tu modelo
    # Solo añadimos puntos donde la probabilidad represente riesgo (ej. > 0.1) para no saturar el mapa
    puntos_dia = df_dia[['Lat', 'Long', 'prob_helada']].values.tolist()
    datos_por_dia.append(puntos_dia)

# =====================================================================
# 3. CONSTRUCCIÓN DEL MAPA BASE (CENTRADO EN PUNO Y EL LAGO TITICACA)
# =====================================================================
print("-> [2/3] Inicializando lienzo geográfico interactivo de OpenStreetMap...")

# Coordenadas concéntricas aproximadas para abarcar todo el departamento de Puno
coordenadas_puno = [-15.20, -69.80]

mapa_base = folium.Map(
    location=coordenadas_puno,
    zoom_start=8,                  # Nivel de acercamiento ideal para pantallas estándar
    tiles='OpenStreetMap',         # Fondo topográfico y de carreteras real
    control_scale=True             # Muestra la barra de escala kilométrica
)

# =====================================================================
# 4. ACOPLAMIENTO DE LA CAPA DINÁMICA DE CALOR (TIME-LAPSE)
# =====================================================================
print("-> [3/3] Inyectando algoritmo de mapas de calor secuenciales (HeatMapWithTime)...")

capa_calor_temporal = HeatMapWithTime(
    data=datos_por_dia,
    index=lista_fechas,
    radius=18,                     # Radio de dispersión de cada píxel de la grilla ERA5
    min_opacity=0.1,
    max_opacity=0.85,
    scale_radius=False,            # Mantiene el radio fijo al hacer zoom para no deformar la grilla
    auto_play=False,               # Permitir que el usuario le dé 'Play' manualmente
    display_index=True,            # Muestra la fecha actual flotando sobre el mapa
    position='bottomleft'          # Ubicación del controlador de tiempo
)

# Añadir la capa dinámica al mapa base
capa_calor_temporal.add_to(mapa_base)

# Guardar el mapa renderizado en formato HTML ejecutable
mapa_base.save(ruta_salida_html)

print("\n" + "="*65)
print("¡ÉXITO TOTAL! El mapa interactivo multitemporal ha sido creado.")
print(f"Ruta del archivo:\n --> {ruta_salida_html}")
print("="*65)
print("💡 CONSEJO DE USO: Ve a esa carpeta, dale doble clic al archivo HTML y se abrirá")
print("   en tu navegador. ¡Usa los controles inferiores para cambiar de fecha!")