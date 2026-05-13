import pandas as pd
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

# Cargar DEM desde CSV comprimido
print("Cargando DEM liviano...")
df = pd.read_csv('dem_puno.csv.gz')

# Extraer coordenadas
longitudes = df['longitud'].unique()
latitudes = df['latitud'].unique()
elevacion = df['elevacion'].values.reshape(len(latitudes), len(longitudes))

# Normalizar para OpenGL
min_elev, max_elev = elevacion.min(), elevacion.max()
Z = (elevacion - min_elev) / (max_elev - min_elev) * 50 - 25  # Escala -25 a 25

# Crear malla X, Y
ny, nx = elevacion.shape
X = np.linspace(-40, 40, nx)
Y = np.linspace(-40, 40, ny)
X, Y = np.meshgrid(X, Y)

print(f"Malla lista: {nx} x {ny} = {nx*ny} vértices")
print(f"Altitud min: {min_elev:.0f}m, max: {max_elev:.0f}m")

# Ahora usa X, Y, Z para tu visualización 3D