import rasterio
import matplotlib.pyplot as plt

# Cambia por el nombre de UNO de tus archivos DEM
archivo_dem = "s15_w069_1arc_v3.tif"  # <- pon el nombre correcto

# Cargar y leer
with rasterio.open(archivo_dem) as src:
    dem = src.read(1)  # Matriz de elevaciones
    print(f"Dimensiones: {dem.shape}")
    print(f"Altitud min: {dem.min():.0f} m")
    print(f"Altitud max: {dem.max():.0f} m")
    print(f"Altitud media: {dem.mean():.0f} m")
    
    # Mostrar imagen
    plt.figure(figsize=(10, 8))
    plt.imshow(dem, cmap='terrain')
    plt.colorbar(label='Elevación (m)')
    plt.title(f'DEM - {archivo_dem}')
    plt.savefig('dem_visualizado.png')
    plt.show()