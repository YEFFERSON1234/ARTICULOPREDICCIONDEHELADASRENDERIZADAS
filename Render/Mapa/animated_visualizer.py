# Render/Mapa/visualizadoranimado.py (versión corregida)
import pandas as pd
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import os
import glob
from scipy.interpolate import griddata

print("="*60)
print("VISUALIZADOR 3D ANIMADO - RIESGO DE HELADAS EN EL ALTIPLANO")
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
# 2. FILTRAR TERRENO
# ==========================================
print("\n2. Procesando terreno...")
df_terreno_filtrado = df_terreno[
    (df_terreno['longitud'] >= -71.5) & (df_terreno['longitud'] <= -68.5) &
    (df_terreno['latitud'] >= -17.5) & (df_terreno['latitud'] <= -14.5)
]

longitudes = np.sort(df_terreno_filtrado['longitud'].unique())
latitudes = np.sort(df_terreno_filtrado['latitud'].unique())
print(f"   Malla original: {len(longitudes)} x {len(latitudes)}")

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
# 3. NORMALIZAR ELEVACIÓN
# ==========================================
print("\n3. Normalizando elevación...")
min_elev, max_elev = np.nanmin(elevacion), np.nanmax(elevacion)
print(f"   Altitud real: {min_elev:.0f} - {max_elev:.0f} m")

Z = (elevacion - min_elev) / (max_elev - min_elev) * 60 - 30
Z = np.nan_to_num(Z, nan=-10)

X = np.linspace(-40, 40, len(longitudes_filt))
Y = np.linspace(-40, 40, len(latitudes_filt))
X, Y = np.meshgrid(X, Y)

# Reducir resolución
factor = 2
if factor > 1:
    Z = Z[::factor, ::factor]
    X = X[::factor, ::factor]
    Y = Y[::factor, ::factor]
    ny, nx = Z.shape
    print(f"   Resolución reducida: {nx} x {ny}")

# ==========================================
# 4. CARGAR ARCHIVOS DE PREDICCIÓN
# ==========================================
proyecto_root = os.path.dirname(os.path.dirname(script_dir))
pred_folder = os.path.join(proyecto_root, 'limpiezadedatos', 'predicciones_temporales')
archivos_pred = sorted(glob.glob(os.path.join(pred_folder, 'pred_*.csv')))

if len(archivos_pred) == 0:
    print(f"\n⚠️ No se encontraron predicciones en {pred_folder}")
    print("Ejecuta primero: python modelos/regenerar_predicciones_con_coords.py")
    sys.exit()

print(f"\n4. Cargando predicciones: {len(archivos_pred)} archivos")

# Variables globales
vertices = None
colores = None
normales = None
frame_actual = 0
total_frames = len(archivos_pred)

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

def generar_malla_desde_riesgo(riesgo_malla):
    global vertices, colores, normales
    vertices = []
    colores = []
    normales = []
    
    for i in range(ny - 1):
        for j in range(nx - 1):
            h1, h2, h3, h4 = Z[i, j], Z[i+1, j], Z[i+1, j+1], Z[i, j+1]
            x1, x2, x3, x4 = X[i, j], X[i+1, j], X[i+1, j+1], X[i, j+1]
            y1, y2, y3, y4 = Y[i, j], Y[i+1, j], Y[i+1, j+1], Y[i, j+1]
            
            r = (riesgo_malla[i, j] + riesgo_malla[i+1, j] + riesgo_malla[i+1, j+1] + riesgo_malla[i, j+1]) / 4
            
            if r > 0.7:
                color = (1.0, 0.1, 0.1)
            elif r > 0.5:
                color = (1.0, 0.4, 0.0)
            elif r > 0.3:
                color = (1.0, 0.8, 0.0)
            elif r > 0.1:
                color = (0.2, 0.7, 0.2)
            else:
                color = (0.2, 0.4, 0.8)
            
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

def cargar_frame(indice):
    global frame_actual
    archivo = archivos_pred[indice]
    df_pred = pd.read_csv(archivo)
    
    fecha_str = os.path.basename(archivo).replace('pred_', '').replace('.csv', '')
    fecha_formateada = f"{fecha_str[:4]}-{fecha_str[4:6]}-{fecha_str[6:8]}"
    print(f"   📅 Frame {indice+1}/{total_frames}: {fecha_formateada}")
    
    # Verificar columnas
    if 'longitud' not in df_pred.columns or 'latitud' not in df_pred.columns:
        print(f"   ⚠️ Error: El archivo no tiene coordenadas. Regenerar predicciones.")
        return fecha_formateada
    
    # Escalar riesgo
    if 'frost_proba' in df_pred.columns:
        riesgo_vals = df_pred['frost_proba'].values
    else:
        riesgo_vals = np.random.uniform(0.1, 0.6, len(df_pred))
    
    riesgo_min, riesgo_max = riesgo_vals.min(), riesgo_vals.max()
    if riesgo_max > riesgo_min:
        riesgo_escalado = (riesgo_vals - riesgo_min) / (riesgo_max - riesgo_min)
    else:
        riesgo_escalado = riesgo_vals
    
    # Interpolar
    puntos_pred = np.column_stack((df_pred['longitud'].values, df_pred['latitud'].values, riesgo_escalado))
    X_malla, Y_malla = np.meshgrid(longitudes_filt, latitudes_filt)
    
    try:
        riesgo_interpolado = griddata(
            puntos_pred[:, :2], puntos_pred[:, 2], 
            (X_malla, Y_malla), method='linear', fill_value=0.3
        )
    except (ValueError, IndexError) as e:
        print(f"   [WARNING] Interpolación falló, usando valor por defecto: {e}")
        riesgo_interpolado = np.full((len(latitudes_filt), len(longitudes_filt)), 0.3)
    
    if factor > 1:
        riesgo_interpolado = riesgo_interpolado[::factor, ::factor]
    
    generar_malla_desde_riesgo(riesgo_interpolado)
    frame_actual = indice
    
    return fecha_formateada

# ==========================================
# 5. INICIALIZAR OPENGL
# ==========================================
pygame.init()
display = (1024, 768)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption('Riesgo de Heladas - Altiplano Peruano (Animado)')

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, display[0]/display[1], 0.1, 500)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
glTranslatef(0.0, -20.0, -130.0)
glRotatef(20, 1, 0, 0)
glRotatef(-30, 0, 1, 0)

glEnable(GL_DEPTH_TEST)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glEnable(GL_COLOR_MATERIAL)
glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

glLightfv(GL_LIGHT0, GL_POSITION, [50, 100, 50, 1])
glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.9, 1])

glEnable(GL_CULL_FACE)
glCullFace(GL_BACK)

# Variables de cámara y animación
rot_x, rot_y = 20, -30
zoom = -130
mouse_down = False
last_mouse = (0, 0)
auto_reproducir = False
ultimo_frame_time = pygame.time.get_ticks()
intervalo_ms = 2000

# Cargar primer frame
print("\n5. Cargando primer frame...")
cargar_frame(0)
print(f"   ✅ Vértices: {len(vertices)//3:,}")

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
    global rot_x, rot_y, zoom, mouse_down, last_mouse, auto_reproducir, ultimo_frame_time, frame_actual
    
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
            if event.key == pygame.K_SPACE:
                nuevo = (frame_actual + 1) % total_frames
                cargar_frame(nuevo)
            if event.key == pygame.K_r:
                auto_reproducir = not auto_reproducir
                print(f"🎬 Auto: {'ON' if auto_reproducir else 'OFF'}")
                ultimo_frame_time = pygame.time.get_ticks()
            if event.key == pygame.K_HOME:
                cargar_frame(0)
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_down = True
            last_mouse = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouse_down = False
        if event.type == pygame.MOUSEMOTION and mouse_down:
            x, y = pygame.mouse.get_pos()
            rot_y += (x - last_mouse[0]) * 0.5
            rot_x += (y - last_mouse[1]) * 0.5
            last_mouse = (x, y)
    
    if auto_reproducir:
        ahora = pygame.time.get_ticks()
        if ahora - ultimo_frame_time >= intervalo_ms:
            nuevo = (frame_actual + 1) % total_frames
            cargar_frame(nuevo)
            ultimo_frame_time = ahora
    
    return True

print("\n" + "="*60)
print("VISUALIZADOR 3D ANIMADO CON ILUMINACIÓN")
print("="*60)
print("Leyenda: 🔴 Rojo=Alto 🟠Naranja=Medio-Alto 🟡Amarillo=Medio 🟢Verde=Bajo 🔵Azul=Sin riesgo")
print("Controles: Mouse, Flechas, +/- , ESPACIO(siguiente), R(auto), HOME(inicio), ESC(salir)")
print("="*60)

clock = pygame.time.Clock()
running = True
while running:
    running = manejar_eventos()
    dibujar()
    clock.tick(60)

pygame.quit()
sys.exit()