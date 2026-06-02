import pandas as pd
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

print("="*50)
print("VISUALIZADOR 3D - ALTIPLANO PERUANO")
print("="*50)

# ==========================================
# 1. CARGAR DEM DESDE CSV COMPRIMIDO
# ==========================================
print("Cargando DEM liviano desde CSV...")
df = pd.read_csv(r'Render\Mapa\dem_puno_render.csv.gz')

print(f"Filas cargadas: {len(df):,}")
print(f"Altitud min: {df['elevacion'].min():.0f}m")
print(f"Altitud max: {df['elevacion'].max():.0f}m")

# ==========================================
# 2. CREAR MALLA REGULAR A PARTIR DE LOS DATOS
# ==========================================
# Encontrar las coordenadas únicas ordenadas
longitudes = np.sort(df['longitud'].unique())
latitudes = np.sort(df['latitud'].unique())

print(f"Longitudes únicas: {len(longitudes)}")
print(f"Latitudes únicas: {len(latitudes)}")

# Crear matriz de ceros para la elevación
elevacion = np.zeros((len(latitudes), len(longitudes)))

# Llenar la matriz con los valores del CSV
# Crear un diccionario para acceso rápido
elev_dict = {}
for _, row in df.iterrows():
    key = (row['latitud'], row['longitud'])
    elev_dict[key] = row['elevacion']

# Llenar la matriz
for i, lat in enumerate(latitudes):
    for j, lon in enumerate(longitudes):
        key = (lat, lon)
        if key in elev_dict:
            elevacion[i, j] = elev_dict[key]
        else:
            elevacion[i, j] = np.nan

print(f"Matriz creada: {elevacion.shape[0]} x {elevacion.shape[1]}")

# Eliminar filas/columnas con muchos NaN
# Encontrar el bounding box válido
filas_validas = ~np.isnan(elevacion).all(axis=1)
cols_validas = ~np.isnan(elevacion).all(axis=0)

elevacion = elevacion[filas_validas][:, cols_validas]
longitudes_filtradas = longitudes[cols_validas]
latitudes_filtradas = latitudes[filas_validas]

print(f"Matriz filtrada: {elevacion.shape[0]} x {elevacion.shape[1]}")

# Interpolar valores NaN restantes (opcional)
if np.isnan(elevacion).any():
    print("Interpolando valores faltantes...")
    from scipy import interpolate
    # Crear máscara
    mask = ~np.isnan(elevacion)
    xx, yy = np.meshgrid(np.arange(elevacion.shape[1]), np.arange(elevacion.shape[0]))
    elevacion = interpolate.griddata(
        (xx[mask], yy[mask]), elevacion[mask], (xx, yy), method='nearest'
    )

# ==========================================
# 3. NORMALIZAR PARA OPENGL
# ==========================================
min_elev, max_elev = np.nanmin(elevacion), np.nanmax(elevacion)
Z = (elevacion - min_elev) / (max_elev - min_elev) * 40 - 20  # Escala -20 a 20

# Crear malla X, Y (normalizadas para OpenGL)
ny, nx = elevacion.shape
X = np.linspace(-40, 40, nx)
Y = np.linspace(-40, 40, ny)
X, Y = np.meshgrid(X, Y)

print(f"Malla final: {nx} x {ny} = {nx*ny} vértices")
print(f"Altitud min: {min_elev:.0f}m, max: {max_elev:.0f}m")

# Si la malla es muy grande, reducir resolución
max_vertices = 50000  # Límite para rendimiento
if nx * ny > max_vertices:
    factor = int(np.sqrt((nx * ny) / max_vertices)) + 1
    print(f"Reduciendo resolución por factor {factor}...")
    Z = Z[::factor, ::factor]
    X = X[::factor, ::factor]
    Y = Y[::factor, ::factor]
    ny, nx = Z.shape
    print(f"Nueva malla: {nx} x {ny} = {nx*ny} vértices")

# ==========================================
# 4. GENERAR VÉRTICES Y COLORES
# ==========================================
print("Generando vértices y colores...")
vertices = []
colores = []

for i in range(ny - 1):
    for j in range(nx - 1):
        # Alturas de los 4 puntos
        h1, h2, h3, h4 = Z[i, j], Z[i+1, j], Z[i+1, j+1], Z[i, j+1]
        
        # Coordenadas X, Y
        x1, x2, x3, x4 = X[i, j], X[i+1, j], X[i+1, j+1], X[i, j+1]
        y1, y2, y3, y4 = Y[i, j], Y[i+1, j], Y[i+1, j+1], Y[i, j+1]
        
        # Color basado en altitud (rojo=alto, azul=bajo)
        riesgo = (h1 + 20) / 40
        color_r = min(1.0, riesgo * 1.5)
        color_g = min(1.0, 1 - riesgo * 0.5)
        color_b = min(1.0, 1 - riesgo)
        
        # Triángulo 1
        vertices.extend([x1, h1, y1])
        vertices.extend([x2, h2, y2])
        vertices.extend([x3, h3, y3])
        for _ in range(3):
            colores.extend([color_r, color_g, color_b])
        
        # Triángulo 2
        vertices.extend([x1, h1, y1])
        vertices.extend([x3, h3, y3])
        vertices.extend([x4, h4, y4])
        for _ in range(3):
            colores.extend([color_r, color_g, color_b])

vertices = np.array(vertices, dtype=np.float32)
colores = np.array(colores, dtype=np.float32)
print(f"Vértices generados: {len(vertices)//3:,}")
print(f"Triángulos: {len(vertices)//9:,}")

# ==========================================
# 5. INICIALIZAR PYGAME Y OPENGL
# ==========================================
pygame.init()
display = (1024, 768)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption('Mapa de Heladas - Altiplano Peruano')

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, display[0]/display[1], 0.1, 200)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
glTranslatef(0.0, -15.0, -80.0)
glRotatef(25, 1, 0, 0)
glRotatef(-40, 0, 1, 0)

glEnable(GL_DEPTH_TEST)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [0, 100, 50, 1])
glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1])

# Variables de cámara
rot_x, rot_y = 25, -40
zoom = -80
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
    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glColorPointer(3, GL_FLOAT, 0, colores)
    glDrawArrays(GL_TRIANGLES, 0, len(vertices)//3)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    
    pygame.display.flip()

def manejar_eventos():
    global rot_x, rot_y, zoom, mouse_down, last_mouse
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            elif event.key == pygame.K_UP:
                rot_x -= 5
            elif event.key == pygame.K_DOWN:
                rot_x += 5
            elif event.key == pygame.K_LEFT:
                rot_y -= 5
            elif event.key == pygame.K_RIGHT:
                rot_y += 5
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                zoom += 10
            elif event.key == pygame.K_MINUS:
                zoom -= 10
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            global mouse_down, last_mouse
            mouse_down = True
            last_mouse = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouse_down = False
        elif event.type == pygame.MOUSEMOTION and mouse_down:
            x, y = pygame.mouse.get_pos()
            rot_y += (x - last_mouse[0]) * 0.5
            rot_x += (y - last_mouse[1]) * 0.5
            last_mouse = (x, y)
    return True

print("\n" + "="*50)
print("VISUALIZADOR 3D INICIADO")
print("="*50)
print("Controles:")
print("  - Mouse arrastrar: Rotar vista")
print("  - Flechas: Rotar")
print("  - +/- : Zoom")
print("  - ESC: Salir")
print("="*50)

clock = pygame.time.Clock()
running = True
while running:
    running = manejar_eventos()
    dibujar()
    clock.tick(60)

pygame.quit()
sys.exit()