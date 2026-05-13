import rasterio
import pandas as pd
import numpy as np
import os
import glob

print("="*50)
print("CONVIRTIENDO TILES DEM A CSV COMPRIMIDO")
print("="*50)

# Carpeta con los DEM
carpeta_dems = "Archivos.tiff-renderizar"

# Verificar que la carpeta existe
if not os.path.exists(carpeta_dems):
    print(f"ERROR: La carpeta '{carpeta_dems}' no existe")
    print("Creándola...")
    os.makedirs(carpeta_dems)
    print(f"Por favor, coloca tus archivos .tif en la carpeta '{carpeta_dems}'")
    exit()

# Buscar todos los archivos .tif (ajusta el patrón según tu nombre real)
archivos_tif = glob.glob(os.path.join(carpeta_dems, "*.tif"))
print(f"Encontrados {len(archivos_tif)} archivos DEM")

if len(archivos_tif) == 0:
    print("No se encontraron archivos .tif")
    print("Buscando con otro patrón...")
    archivos_tif = glob.glob(os.path.join(carpeta_dems, "*.tiff"))
    print(f"Encontrados {len(archivos_tif)} archivos .tiff")
    
if len(archivos_tif) == 0:
    print("No hay archivos DEM para procesar")
    exit()

# Lista para guardar todos los datos
todas_las_elevaciones = []

for archivo in archivos_tif:
    nombre = os.path.basename(archivo)
    print(f"Procesando: {nombre}")
    
    with rasterio.open(archivo) as src:
        dem = src.read(1)
        bounds = src.bounds
        
        # Reducir resolución (factor: más alto = menos peso)
        factor = 10
        dem_reducido = dem[::factor, ::factor]
        
        ny, nx = dem_reducido.shape
        print(f"  Dimensiones reducidas: {nx} x {ny}")
        
        # Crear coordenadas
        longitudes = np.linspace(bounds.left, bounds.right, nx)
        latitudes = np.linspace(bounds.top, bounds.bottom, ny)
        
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        
        # Crear DataFrame para este tile
        df_tile = pd.DataFrame({
            'longitud': lon_grid.flatten(),
            'latitud': lat_grid.flatten(),
            'elevacion': dem_reducido.flatten()
        })
        
        # Filtrar valores válidos
        df_tile = df_tile.dropna()
        df_tile = df_tile[df_tile['elevacion'] > -1000]
        
        print(f"  Puntos válidos: {len(df_tile)}")
        todas_las_elevaciones.append(df_tile)

# Unir todos los tiles
print("\nUniendo todos los tiles...")
df = pd.concat(todas_las_elevaciones, ignore_index=True)

# Guardar como CSV comprimido
archivo_salida = 'dem_puno_render.csv.gz'
df.to_csv(archivo_salida, compression='gzip', index=False)

# Mostrar resultados
import os
tamaño = os.path.getsize(archivo_salida) / (1024 * 1024)

print("\n" + "="*50)
print("CONVERSIÓN COMPLETADA")
print("="*50)
print(f"✅ Archivo guardado: {archivo_salida}")
print(f"📊 Filas totales: {len(df):,}")
print(f"💾 Tamaño del archivo: {tamaño:.2f} MB")
print(f"🏔️ Altitud min: {df['elevacion'].min():.0f} m")
print(f"🏔️ Altitud max: {df['elevacion'].max():.0f} m")import rasterio
import pandas as pd
import numpy as np
import os
import glob

print("="*50)
print("CONVIRTIENDO TILES DEM A CSV COMPRIMIDO")
print("="*50)

# Carpeta con los DEM
carpeta_dems = "Archivos.tiff-renderizar"

# Verificar que la carpeta existe
if not os.path.exists(carpeta_dems):
    print(f"ERROR: La carpeta '{carpeta_dems}' no existe")
    print("Creándola...")
    os.makedirs(carpeta_dems)
    print(f"Por favor, coloca tus archivos .tif en la carpeta '{carpeta_dems}'")
    exit()

# Buscar todos los archivos .tif (ajusta el patrón según tu nombre real)
archivos_tif = glob.glob(os.path.join(carpeta_dems, "*.tif"))
print(f"Encontrados {len(archivos_tif)} archivos DEM")

if len(archivos_tif) == 0:
    print("No se encontraron archivos .tif")
    print("Buscando con otro patrón...")
    archivos_tif = glob.glob(os.path.join(carpeta_dems, "*.tiff"))
    print(f"Encontrados {len(archivos_tif)} archivos .tiff")
    
if len(archivos_tif) == 0:
    print("No hay archivos DEM para procesar")
    exit()

# Lista para guardar todos los datos
todas_las_elevaciones = []

for archivo in archivos_tif:
    nombre = os.path.basename(archivo)
    print(f"Procesando: {nombre}")
    
    with rasterio.open(archivo) as src:
        dem = src.read(1)
        bounds = src.bounds
        
        # Reducir resolución (factor: más alto = menos peso)
        factor = 10
        dem_reducido = dem[::factor, ::factor]
        
        ny, nx = dem_reducido.shape
        print(f"  Dimensiones reducidas: {nx} x {ny}")
        
        # Crear coordenadas
        longitudes = np.linspace(bounds.left, bounds.right, nx)
        latitudes = np.linspace(bounds.top, bounds.bottom, ny)
        
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        
        # Crear DataFrame para este tile
        df_tile = pd.DataFrame({
            'longitud': lon_grid.flatten(),
            'latitud': lat_grid.flatten(),
            'elevacion': dem_reducido.flatten()
        })
        
        # Filtrar valores válidos
        df_tile = df_tile.dropna()
        df_tile = df_tile[df_tile['elevacion'] > -1000]
        
        print(f"  Puntos válidos: {len(df_tile)}")
        todas_las_elevaciones.append(df_tile)

# Unir todos los tiles
print("\nUniendo todos los tiles...")
df = pd.concat(todas_las_elevaciones, ignore_index=True)

# Guardar como CSV comprimido
archivo_salida = 'dem_puno_render.csv.gz'
df.to_csv(archivo_salida, compression='gzip', index=False)

# Mostrar resultados
import os
tamaño = os.path.getsize(archivo_salida) / (1024 * 1024)

print("\n" + "="*50)
print("CONVERSIÓN COMPLETADA")
print("="*50)
print(f"✅ Archivo guardado: {archivo_salida}")
print(f"📊 Filas totales: {len(df):,}")
print(f"💾 Tamaño del archivo: {tamaño:.2f} MB")
print(f"🏔️ Altitud min: {df['elevacion'].min():.0f} m")
print(f"🏔️ Altitud max: {df['elevacion'].max():.0f} m")
print(f"🏔️ Altitud media: {df['elevacion'].mean():.0f} m")

print("\n📁 Ya puedes subir este archivo a GitHub:")
print(f"   git add {archivo_salida}")
print("   git commit -m 'Agregar DEM liviano para renderizado'")
print("   git push")
print(f"🏔️ Altitud media: {df['elevacion'].mean():.0f} m")

print("\n📁 Ya puedes subir este archivo a GitHub:")
print(f"   git add {archivo_salida}")
print("   git commit -m 'Agregar DEM liviano para renderizado'")
print("   git push")