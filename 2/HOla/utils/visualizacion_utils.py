# utils/visualizacion_utils.py
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def cargar_y_preparar_dem(ruta_dem, factor_reduccion=8, recorte_borde=50):
    """
    Carga DEM y lo prepara para visualización.
    
    Args:
        ruta_dem (str): Ruta al archivo TIF del DEM
        factor_reduccion (int): Factor de reducción de resolución
        recorte_borde (int): Píxeles a recortar en bordes
    
    Returns:
        tuple: (X, Y, Z, dem_info) - Coordenadas y datos para graficar
    """
    print("   Cargando DEM...")
    with rasterio.open(ruta_dem) as src:
        dem = src.read(1)
        # Guardar metadatos para coordenadas reales
        bounds = src.bounds
        transform = src.transform
    
    # Limpiar valores NoData
    dem = np.where(dem < -1000, np.nan, dem)
    
    # Reducir resolución
    dem_reducido = dem[::factor_reduccion, ::factor_reduccion]
    
    # Recortar bordes con NaN
    if dem_reducido.shape[0] > recorte_borde * 2:
        dem_recortado = dem_reducido[recorte_borde:-recorte_borde, 
                                     recorte_borde:-recorte_borde]
    else:
        dem_recortado = dem_reducido
    
    ny, nx = dem_recortado.shape
    
    # Crear coordenadas geográficas
    longitudes = np.linspace(bounds.left, bounds.right, nx)
    latitudes = np.linspace(bounds.bottom, bounds.top, ny)
    X, Y = np.meshgrid(longitudes, latitudes)
    
    dem_info = {
        'min': np.nanmin(dem_recortado),
        'max': np.nanmax(dem_recortado),
        'media': np.nanmean(dem_recortado),
        'nx': nx,
        'ny': ny
    }
    
    return X, Y, dem_recortado, dem_info


def cargar_predictions_csv(ruta_csv):
    """
    Carga archivo CSV con predicciones reales del modelo.
    
    Args:
        ruta_csv (str): Ruta a predictions.csv con columnas [longitud, latitud, probabilidad]
    
    Returns:
        pd.DataFrame: DataFrame con coordenadas y probabilidad
    """
    print("   Cargando predicciones del modelo...")
    df = pd.read_csv(ruta_csv)
    print(f"   Predicciones cargadas: {len(df)} puntos")
    
    # Verificar columnas necesarias
    columnas_requeridas = ['longitud', 'latitud', 'probabilidad']
    for col in columnas_requeridas:
        if col not in df.columns:
            # Intentar nombres alternativos comunes
            if col == 'probabilidad':
                posibles = ['riesgo', 'prediccion', 'prob_helada', 'helada_prob']
                for p in posibles:
                    if p in df.columns:
                        df['probabilidad'] = df[p]
                        break
                else:
                    raise ValueError(f"No encuentro columna de predicción. Tengo: {df.columns}")
    
    return df


def mapear_prediccion_a_dem(X, Y, df_predicciones):
    """
    Interpola las predicciones (puntos dispersos) a la malla del DEM.
    
    Args:
        X, Y: Mallas de coordenadas del DEM
        df_predicciones: DataFrame con columnas [longitud, latitud, probabilidad]
    
    Returns:
        np.array: Malla 2D con probabilidades interpoladas
    """
    from scipy.interpolate import griddata
    
    puntos = df_predicciones[['longitud', 'latitud']].values
    valores = df_predicciones['probabilidad'].values
    
    # Interpolar a la malla regular del DEM
    riesgo_malla = griddata(puntos, valores, (X, Y), method='linear', fill_value=np.nan)
    
    # Rellenar NaN con vecino más cercano
    from scipy.ndimage import distance_transform_edt
    if np.any(np.isnan(riesgo_malla)):
        mask = np.isnan(riesgo_malla)
        riesgo_malla[mask] = griddata(puntos, valores, (X[mask], Y[mask]), 
                                      method='nearest')
    
    return np.clip(riesgo_malla, 0, 1)


def crear_mapa_heladas(X, Y, Z, riesgo, dem_info, ruta_salida, 
                       titulo="MAPA DE RIESGO DE HELADAS - ALTIPLANO PERUANO",
                       mostrar_stats=True):
    """
    Crea y guarda el mapa 3D completo.
    
    Args:
        X, Y: Coordenadas geográficas
        Z: Elevación
        riesgo: Malla con probabilidades de helada
        dem_info: Diccionario con estadísticas del DEM
        ruta_salida: Dónde guardar la imagen PNG
        titulo: Título del mapa
        mostrar_stats: Si mostrar estadísticas en el mapa
    """
    print("   Generando visualización 3D...")
    
    # Crear colormap
    colores_helada = ['darkblue', 'blue', 'cyan', 'lime', 'yellow', 'orange', 'red', 'darkred']
    cmap_riesgo = LinearSegmentedColormap.from_list('helada', colores_helada, N=256)
    
    # Crear figura
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Superficie con colores de riesgo
    surf = ax.plot_surface(X, Y, Z,
                          facecolors=cmap_riesgo(riesgo),
                          rstride=1, cstride=1,
                          alpha=0.95,
                          linewidth=0,
                          antialiased=True)
    
    # Configurar etiquetas
    ax.set_xlabel('Longitud (°Oeste)', fontsize=12, labelpad=10)
    ax.set_ylabel('Latitud (°Sur)', fontsize=12, labelpad=10)
    ax.set_zlabel('Elevación (msnm)', fontsize=12, labelpad=10)
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
    
    # Ángulo de cámara
    ax.view_init(elev=25, azim=-60)
    
    # Barra de colores
    mappable = plt.cm.ScalarMappable(cmap=cmap_riesgo)
    mappable.set_array(riesgo)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Probabilidad de Helada', fontsize=12, rotation=270, labelpad=20)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['Baja', 'Moderada', 'Media', 'Alta', 'Muy Alta'])
    
    # Estadísticas
    if mostrar_stats:
        stats_text = (f"Altitud mínima: {dem_info['min']:.0f} m\n"
                     f"Altitud máxima: {dem_info['max']:.0f} m\n"
                     f"Altitud media: {dem_info['media']:.0f} m")
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Guardar
    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.savefig(ruta_salida, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Mapa guardado: {ruta_salida}")
    return ruta_salida