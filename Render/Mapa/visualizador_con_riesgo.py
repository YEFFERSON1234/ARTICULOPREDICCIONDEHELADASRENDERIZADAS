import pandas as pd
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import os

print("="*60)
print("VISUALIZADOR 3D - RIESGO DE HELADAS EN EL ALTIPLANO")
print("="*60)

# ==========================================
# 1. CARGAR TERRENO (DEM)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'dem_puno_render.csv.gz')

print("\n1. Cargando terreno desde DEM...")
df_terreno = pd.read_csv(csv_path)
print(f"   Puntos del terreno: {len(df_terreno):,}")

# ==========================================
# 2. CARGAR PREDICCIONES
# ==========================================
proyecto_root = os.path.dirname(os.path.dirname(script_dir))
pred_path = os.path.join(proyecto_root, 'limpiezadedatos', 'predictions.csv')

print(f"\n2. Cargando predicciones...")
df_pred = pd.read_csv(pred_path)
print(f"   Predicciones: {len(df_pred):,}")

# ==========================================
# 3. COORDENADAS DE ESTACIONES
# ==========================================
coordenadas_estaciones = {
    "Puno": (-15.84, -70.02), "Juliaca": (-15.49, -70.13),
    "Azángaro": (-14.90, -70.10), "Ayaviri": (-14.92, -70.59),
    "Macusani": (-14.08, -70.43), "Mazocruz": (-16.75, -69.72),
    "Lampa": (-15.35, -70.37), "Yunguyo": (-16.25, -69.08),
    "Juli": (-16.22, -69.45), "Desaguadero": (-16.57, -69.04),
    "Cojata": (-15.02, -69.37), "Crucero": (-14.35, -70.03),
    "Crucero Alto": (-15.78, -70.92)
}

df_pred['latitud'] = df_pred['station'].map(lambda s: coordenadas_estaciones.get(s, (np.nan, np.nan))[0])
df_pred['longitud'] = df_pred['station'].map(lambda s: coordenadas_estaciones.get(s, (np.nan, np.nan))[1])
df_pred = df_pred.dropna(subset=['latitud', 'longitud'])

if 'frost_proba' not in df_pred.columns:
    df_pred['frost_proba'] = np.random.uniform(0.1, 0.6, len(df_pred))

print(f"   Riesgo original: min={df_pred['frost_proba'].min():.3f}, max={df_pred['frost_proba'].max():.3f}")

# ESCALAR EL RIESGO PARA VER TODOS LOS COLORES
riesgo_min = df_pred['frost_proba'].min()
riesgo_max = df_pred['frost_proba'].max()
df_pred['frost_proba_escalado'] = (df_pred['frost_proba'] - riesgo_min) / (riesgo_max - riesgo_min)
print(f"   Riesgo escalado: min=0.000, max=1.000")

# ==========================================
# 4. FILTRAR Y CREAR MALLA
# ==========================================
print("\n3. Filtrando terreno...")
df_terreno_filtrado = df_terreno[
    (df_terreno['longitud'] >= -71.5) & (df_terreno['longitud'] <= -68.5) &
    (df_terreno['latitud'] >= -17.5) & (df_terreno['latitud'] <= -14.5)
]

longitudes = np.sort(df_terreno_filtrado['longitud'].unique())
latitudes = np.sort(df_terreno_filtrado['latitud'].unique())
print(f"   Malla: {len(longitudes)} x {len(latitudes)}")

ny, nx = len(latitudes), len(longitudes)
elevacion = np.full((ny, nx), np.nan)
elev_dict = {(row['latitud'], row['longitud']): row['elevacion'] for _, row in df_terreno_filtrado.iterrows()}

for i, lat in enumerate(latitudes):
    for j, lon in enumerate(longitudes):
        elevacion[i, j] = elev_dict.get((lat, lon), np.nan)

# Limpiar NaN
filas_validas = ~np.isnan(elevacion).all(axis=1)
cols_validas = ~np.isnan(elevacion).all(axis=0)
elevacion = elevacion[filas_validas][:, cols_validas]
longitudes_filt = longitudes[cols_validas]
latitudes_filt = latitudes[filas_validas]
print(f"   Matriz final: {elevacion.shape[0]} x {elevacion.shape[1]}")

# ==========================================
# 5. INTERPOLAR RIESGO ESCALADO
# ==========================================
print("\n4. Interpolando riesgo...")
from scipy.interpolate import griddata

puntos_pred = df_pred[['longitud', 'latitud', 'frost_proba_escalado']].values
X_malla, Y_malla = np.meshgrid(longitudes_filt, latitudes_filt)

riesgo_malla = griddata(
    puntos_pred[:, :2], puntos_pred[:, 2], 
    (X_malla, Y_malla), method='linear', fill_value=0.3
)
print(f"   Riesgo interpolado: min={riesgo_malla.min():.3f}, max={riesgo_malla.max():.3f}")

# ==========================================
# 6. NORMALIZAR ELEVACIÓN (MÁS DRAMÁTICA)
# ==========================================
print("\n5. Normalizando elevación...")
min_elev, max_elev = np.nanmin(elevacion), np.nanmax(elevacion)
print(f"   Altitud real: {min_elev:.0f} - {max_elev:.0f} m")

# ESCALA DE ALTURA MÁS DRAMÁTICA (mejora el 3D)
Z = (elevacion - min_elev) / (max_elev - min_elev) * 60 - 30  # Rango -30 a 30
Z = np.nan_to_num(Z, nan=-10)

X = np.linspace(-40, 40, len(longitudes_filt))
Y = np.linspace(-40, 40, len(latitudes_filt))
X, Y = np.meshgrid(X, Y)

# Reducir resolución para mejor rendimiento
factor = 2
if factor > 1:
    Z = Z[::factor, ::factor]
    X = X[::factor, ::factor]
    Y = Y[::factor, ::factor]
    riesgo_malla = riesgo_malla[::factor, ::factor]
    ny, nx = Z.shape
    print(f"   Resolución reducida: {nx} x {ny}")

# ==========================================
# 7. GENERAR VÉRTICES CON COLORES Y NORMALES
# ==========================================
print("\n6. Generando malla 3D...")

vertices = []
colores = []
normales = []

def calcular_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    ux, uy, uz = x2 - x1, y2 - y1, z2 - z1
    vx, vy, vz = x3 - x1, y3 - y1, z3 - z1
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = np.sqrt(nx*nx + ny*ny + nz*nz)
    if length > 0:
        return nx/length, ny/length, nz/length
    return 0, 1, 0

for i in range(ny - 1):
    for j in range(nx - 1):
        h1, h2, h3, h4 = Z[i, j], Z[i+1, j], Z[i+1, j+1], Z[i, j+1]
        x1, x2, x3, x4 = X[i, j], X[i+1, j], X[i+1, j+1], X[i, j+1]
        y1, y2, y3, y4 = Y[i, j], Y[i+1, j], Y[i+1, j+1], Y[i, j+1]
        
        # Riesgo promedio
        r = (riesgo_malla[i, j] + riesgo_malla[i+1, j] + riesgo_malla[i+1, j+1] + riesgo_malla[i, j+1]) / 4
        
        # Color según riesgo (más vibrante)
        if r > 0.7:
            color = (1.0, 0.1, 0.1)   # Rojo intenso
        elif r > 0.5:
            color = (1.0, 0.4, 0.0)   # Naranja
        elif r > 0.3:
            color = (1.0, 0.8, 0.0)   # Amarillo
        elif r > 0.1:
            color = (0.2, 0.7, 0.2)   # Verde
        else:
            color = (0.2, 0.4, 0.8)   # Azul
        
        # Triángulo 1
        n1 = calcular_normal(x1, h1, y1, x2, h2, y2, x3, h3, y3)
        vertices.extend([x1, h1, y1, x2, h2, y2, x3, h3, y3])
        colores.extend([color[0], color[1], color[2]] * 3)
        normales.extend([n1[0], n1[1], n1[2]] * 3)
        
        # Triángulo 2
        n2 = calcular_normal(x1, h1, y1, x3, h3, y3, x4, h4, y4)
        vertices.extend([x1, h1, y1, x3, h3, y3, x4, h4, y4])
        colores.extend([color[0], color[1], color[2]] * 3)
        normales.extend([n2[0], n2[1], n2[2]] * 3)

vertices = np.array(vertices, dtype=np.float32)
colores = np.array(colores, dtype=np.float32)
normales = np.array(normales, dtype=np.float32)
print(f"   Vértices: {len(vertices)//3:,}")

# ==========================================
# 8. INICIALIZAR OPENGL CON ILUMINACIÓN
# ==========================================
pygame.init()
display = (1024, 768)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption('Riesgo de Heladas - Altiplano Peruano')

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, display[0]/display[1], 0.1, 500)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
glTranslatef(0.0, -20.0, -130.0)
glRotatef(20, 1, 0, 0)
glRotatef(-30, 0, 1, 0)

# CONFIGURACIÓN CON ILUMINACIÓN + COLORES
glEnable(GL_DEPTH_TEST)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glEnable(GL_COLOR_MATERIAL)  # CLAVE: permite que los colores interactúen con la luz
glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

# Configurar luz
glLightfv(GL_LIGHT0, GL_POSITION, [50, 100, 50, 1])
glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.9, 1])
glLightfv(GL_LIGHT0, GL_SPECULAR, [0.2, 0.2, 0.2, 1])

# Material
glMaterialfv(GL_FRONT, GL_SPECULAR, [0.1, 0.1, 0.1, 1])
glMateriali(GL_FRONT, GL_SHININESS, 10)

glEnable(GL_CULL_FACE)
glCullFace(GL_BACK)

# Variables de cámara
rot_x, rot_y = 20, -30
zoom = -130
mouse_down = False
last_mouse = (0, 0)

def dibujar():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    glTranslatef(0.0, 0.0, zoom)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)
    
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    
    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glColorPointer(3, GL_FLOAT, 0, colores)
    glNormalPointer(GL_FLOAT, 0, normales)
    
    glDrawArrays(GL_TRIANGLES, 0, len(vertices)//3)
    
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    
    pygame.display.flip()

def manejar_eventos():
    global rot_x, rot_y, zoom, mouse_down, last_mouse
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_UP:
                rot_x -= 5
            if event.key == pygame.K_DOWN:
                rot_x += 5
            if event.key == pygame.K_LEFT:
                rot_y -= 5
            if event.key == pygame.K_RIGHT:
                rot_y += 5
            if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                zoom += 10
            if event.key == pygame.K_MINUS:
                zoom -= 10
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            global mouse_down, last_mouse
            mouse_down = True
            last_mouse = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouse_down = False
        if event.type == pygame.MOUSEMOTION and mouse_down:
            x, y = pygame.mouse.get_pos()
            rot_y += (x - last_mouse[0]) * 0.5
            rot_x += (y - last_mouse[1]) * 0.5
            last_mouse = (x, y)
    return True

print("\n" + "="*60)
print("VISUALIZADOR 3D CON ILUMINACIÓN")
print("="*60)
print("Leyenda:")
print("  🔴 Rojo    = Riesgo alto")
print("  🟠 Naranja = Riesgo medio-alto")
print("  🟡 Amarillo = Riesgo medio")
print("  🟢 Verde    = Riesgo bajo")
print("  🔵 Azul     = Sin riesgo")
print("="*60)
print("Controles: Mouse arrastrar, Flechas, +/- , ESC")
print("="*60)

clock = pygame.time.Clock()
running = True
while running:
    running = manejar_eventos()
    dibujar()
    clock.tick(60)

pygame.quit()
sys.exit()