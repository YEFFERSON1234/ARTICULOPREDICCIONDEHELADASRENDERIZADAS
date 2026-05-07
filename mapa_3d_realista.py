import numpy as np
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

print("="*50)
print("MAPA 3D REALISTA - ALTIPLANO PERUANO")
print("="*50)

# 1. Cargar DEM
print("\n1. Cargando DEM...")
with rasterio.open('dem_puno_completo.tif') as src:
    dem = src.read(1)

# Reemplazar valores NoData (-32768) con NaN
dem = np.where(dem < -1000, np.nan, dem)

# Reducir resolución (más rápido)
factor = 8
dem_reducido = dem[::factor, ::factor]

# Recortar área de interés (evitar bordes con NaN)
dem_recortado = dem_reducido[50:-50, 50:-50] if dem_reducido.shape[0] > 100 else dem_reducido

ny, nx = dem_recortado.shape
print(f"  Dimensiones: {nx} x {ny}")
print(f"  Altitud min: {np.nanmin(dem_recortado):.0f} m")
print(f"  Altitud max: {np.nanmax(dem_recortado):.0f} m")
print(f"  Altitud media: {np.nanmean(dem_recortado):.0f} m")

# 2. Crear coordenadas geográficas
# Ajustar según tus coordenadas reales
longitudes = np.linspace(-71, -68, nx)   # Longitud Oeste
latitudes = np.linspace(-17, -14, ny)    # Latitud Sur
X, Y = np.meshgrid(longitudes, latitudes)
Z = dem_recortado

# 3. Simular riesgo de helada (después vendrá de predictions.csv)
# Usar altitud como proxy: a mayor altitud, mayor riesgo
riesgo = (Z - np.nanmin(Z)) / (np.nanmax(Z) - np.nanmin(Z))
riesgo = np.clip(riesgo, 0, 1)

# 4. Crear mapa de colores personalizado
# Azul (seguro) -> Amarillo (alerta) -> Rojo (peligro)
colores_helada = ['darkblue', 'blue', 'cyan', 'lime', 'yellow', 'orange', 'red', 'darkred']
cmap_riesgo = LinearSegmentedColormap.from_list('helada', colores_helada, N=256)

# 5. Configurar figura 3D
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# 6. Graficar superficie con color según riesgo
print("\n2. Generando superficie 3D...")
surf = ax.plot_surface(X, Y, Z, 
                       facecolors=cmap_riesgo(riesgo),
                       rstride=1, cstride=1,
                       alpha=0.95,
                       linewidth=0,
                       antialiased=True)

# 7. Configurar etiquetas
ax.set_xlabel('Longitud (°Oeste)', fontsize=12, labelpad=10)
ax.set_ylabel('Latitud (°Sur)', fontsize=12, labelpad=10)
ax.set_zlabel('Elevación (msnm)', fontsize=12, labelpad=10)
ax.set_title('MAPA DE RIESGO DE HELADAS - ALTIPLANO PERUANO\nRegión Puno', 
             fontsize=16, fontweight='bold', pad=20)

# 8. Configurar límites y ángulo de vista
ax.set_xlim(-71, -68)
ax.set_ylim(-17, -14)
ax.view_init(elev=25, azim=-60)  # Ángulo de cámara

# 9. Agregar barra de colores
mappable = plt.cm.ScalarMappable(cmap=cmap_riesgo)
mappable.set_array(riesgo)
cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('Probabilidad de Helada', fontsize=12, rotation=270, labelpad=20)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
cbar.set_ticklabels(['Baja', 'Moderada', 'Media', 'Alta', 'Muy Alta'])

# 10. Agregar texto con estadísticas
stats_text = f"Altitud mínima: {np.nanmin(Z):.0f} m\nAltitud máxima: {np.nanmax(Z):.0f} m\nAltitud media: {np.nanmean(Z):.0f} m"
ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
          fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 11. Guardar figura
plt.tight_layout()
plt.savefig('mapa_heladas_3d_realista.png', dpi=200, bbox_inches='tight')
print("\n3. Imagen guardada: mapa_heladas_3d_realista.png")

# 12. Mostrar
print("\n4. Mostrando mapa...")
plt.show()

print("\n✅ Listo!")