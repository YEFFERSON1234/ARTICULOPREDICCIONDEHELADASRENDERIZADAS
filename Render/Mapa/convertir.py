import rasterio
import pandas as pd
import numpy as np

print("Cargando DEM original...")
with rasterio.open('dem_puno_completo.tif') as src:
    dem = src.read(1)
    bounds = src.bounds

# Reducir resolución (factor = 20 para ~2 MB)
factor = 20
ny, nx = dem.shape
nueva_ny = ny // factor
nueva_nx = nx // factor

dem_reducido = dem[::factor, ::factor]

# Crear coordenadas
longitudes = np.linspace(bounds.left, bounds.right, nueva_nx)
latitudes = np.linspace(bounds.top, bounds.bottom, nueva_ny)

# Crear malla de coordenadas
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

# Aplanar y crear DataFrame
df = pd.DataFrame({
    'longitud': lon_grid.flatten(),
    'latitud': lat_grid.flatten(),
    'elevacion': dem_reducido.flatten()
})

# Eliminar valores nulos (si los hay)
df = df.dropna()

# Guardar como CSV comprimido
df.to_csv('dem_puno.csv.gz', compression='gzip', index=False)

print(f"\nArchivo guardado: dem_puno.csv.gz")
print(f"Tamaño estimado: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB en memoria")
print(f"En disco comprimido: ~1-3 MB")
print(f"Filas: {len(df)}")
print(f"Columnas: {list(df.columns)}")