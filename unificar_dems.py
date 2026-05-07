import rasterio
from rasterio.merge import merge
import glob
import matplotlib.pyplot as plt
import numpy as np

# 1. Listar todos tus archivos DEM
# El * significa "todos los archivos que coincidan"
archivos = glob.glob("s*_*_1arc_v3.tif")  # ajusta según tu nombre real
print(f"Archivos encontrados: {len(archivos)}")
for f in archivos:
    print(f"  - {f}")

# 2. Abrir todos los archivos
archivos_abiertos = []
for archivo in archivos:
    src = rasterio.open(archivo)
    archivos_abiertos.append(src)
    print(f"Abierto: {archivo}, Dimensiones: {src.shape}")

# 3. Unir (merge) todos los tiles usando la funcion merge de rasterio
# Esto combina los tiles en una sola imagen grande
dem_unido, transform = merge(archivos_abiertos)

print(f"\nDEM unificado - Dimensiones: {dem_unido.shape}")
print(f"Bandas: {dem_unido.shape[0]}")
print(f"Filas: {dem_unido.shape[1]}")
print(f"Columnas: {dem_unido.shape[2]}")

# 4. Obtener metadata (informacion) del primer archivo
perfil = archivos_abiertos[0].profile
# Actualizar con las nuevas dimensiones
perfil.update({
    'height': dem_unido.shape[1],
    'width': dem_unido.shape[2],
    'transform': transform
})

# 5. Guardar el DEM unificado
with rasterio.open('dem_puno_completo.tif', 'w', **perfil) as dst:
    dst.write(dem_unido)

print("\n✅ DEM unificado guardado: dem_puno_completo.tif")

# 6. Mostrar estadisticas del DEM unificado
dem_datos = dem_unido[0]  # primera banda
print(f"\n📊 Estadisticas del DEM unificado:")
print(f"  Altitud minima: {np.nanmin(dem_datos):.0f} m")
print(f"  Altitud maxima: {np.nanmax(dem_datos):.0f} m")
print(f"  Altitud media: {np.nanmean(dem_datos):.0f} m")

# 7. Visualizar
plt.figure(figsize=(14, 10))
plt.imshow(dem_datos, cmap='terrain')
plt.colorbar(label='Elevación (m)')
plt.title('DEM Unificado - Región Puno/Altiplano')
plt.xlabel('Columnas (pixeles)')
plt.ylabel('Filas (pixeles)')
plt.savefig('dem_puno_unificado.png', dpi=150)
print("\n✅ Imagen guardada: dem_puno_unificado.png")
plt.show()

# 8. Cerrar todos los archivos
for src in archivos_abiertos:
    src.close()