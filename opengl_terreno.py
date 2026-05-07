import numpy as np
import rasterio
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

print("="*50)
print("OPENGL - MAPA 3D ALTIPLANO")
print("="*50)

# ==========================================
# 1. CARGAR DEM CON RESOLUCIÓN REDUCIDA
# ==========================================
print("Cargando DEM...")
with rasterio.open('dem_puno_completo.tif') as src:
    dem = src.read(1)

# Reemplazar valores inválidos
dem = np.where(dem < -1000, np.nan, dem)

# REDUCIR RESOLUCIÓN DRÁSTICAMENTE (de 10,000 a ~200)
# Esto es clave para que OpenGL funcione
factor = 50  # Ajusta este valor: más alto = más rápido, menos detalle
dem_reducido = dem[::factor, ::factor]

# Recortar bordes con NaN
dem_reducido = dem_reducido[10:-10, 10:-10] if dem_reducido.shape[0] > 20 else dem_reducido

ny, nx = dem_reducido.shape
print(f"Malla reducida: {nx} x {ny} = {nx*ny} vértices")

# Normalizar coordenadas
z_min = np.nanmin(dem_reducido)
z_max = np.nanmax(dem_reducido)
Z = (dem_reducido - z_min) / (z_max - z_min) * 50  # Escala 0-50

# Reemplazar NaN con 0
Z = np.nan_to_num(Z, 0)

# Coordenadas X, Y (escala -40 a 40)
x = np.linspace(-40, 40, nx)
y = np.linspace(-40, 40, ny)
X, Y = np.meshgrid(x, y)

# Riesgo (basado en altitud para simular)
riesgo = Z / 50  # Normalizado 0-1

print(f"Dimensiones finales: {nx} x {ny}")
print(f"Altitud min: {z_min:.0f}m, max: {z_max:.0f}m")

# ==========================================
# 2. INICIALIZAR PYGAME Y OPENGL
# ==========================================
pygame.init()
display = (1024, 768)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption('Mapa de Heladas - Altiplano Peruano')

# Configurar cámara CORRECTAMENTE
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()

# Posición inicial de cámara (más lejos y mejor ángulo)
glTranslatef(0.0, -30.0, -120.0)
glRotatef(25, 1, 0, 0)  # Rotación en X
glRotatef(-45, 0, 1, 0)  # Rotación en Y

# Luces
glEnable(GL_DEPTH_TEST)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [0, 100, 50, 1])
glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])

# Variables de cámara
rot_x = 25
rot_y = -45
zoom = -120
mouse_down = False
last_mouse = (0, 0)

# ==========================================
# 3. GENERAR VÉRTICES Y COLORES (OPTIMIZADO)
# ==========================================
print("Generando malla 3D...")

vertices = []
colores = []

for i in range(ny - 1):
    for j in range(nx - 1):
        # Alturas
        h1 = Z[i, j]
        h2 = Z[i+1, j]
        h3 = Z[i+1, j+1]
        h4 = Z[i, j+1]
        
        # Riesgo promedio
        r_avg = (riesgo[i, j] + riesgo[i+1, j] + riesgo[i+1, j+1] + riesgo[i, j+1]) / 4
        
        # Color según riesgo
        if r_avg > 0.7:
            # Rojo intenso (peligro alto)
            color = (1.0, 0.1, 0.1)
        elif r_avg > 0.5:
            # Naranja (alerta)
            color = (1.0, 0.6, 0.1)
        elif r_avg > 0.3:
            # Amarillo/Verde (moderado)
            color = (0.5, 0.8, 0.2)
        elif r_avg > 0.1:
            # Verde claro (bajo)
            color = (0.2, 0.6, 0.3)
        else:
            # Azul (seguro)
            color = (0.2, 0.4, 0.8)
        
        # Triángulo 1
        vertices.extend([X[i, j], h1, Y[i, j]])
        vertices.extend([X[i+1, j], h2, Y[i+1, j]])
        vertices.extend([X[i+1, j+1], h3, Y[i+1, j+1]])
        colores.extend([color, color, color])
        
        # Triángulo 2
        vertices.extend([X[i, j], h1, Y[i, j]])
        vertices.extend([X[i+1, j+1], h3, Y[i+1, j+1]])
        vertices.extend([X[i, j+1], h4, Y[i, j+1]])
        colores.extend([color, color, color])

vertices = np.array(vertices, dtype=np.float32)
colores = np.array(colores, dtype=np.float32)

print(f"Vértices generados: {len(vertices)//3}")
print(f"Triángulos: {len(vertices)//9}")

# ==========================================
# 4. FUNCIONES DE RENDERIZADO
# ==========================================
def dibujar():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Aplicar transformaciones de cámara
    glTranslatef(0.0, 0.0, zoom)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)
    
    # Dibujar con colores
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
            print(f"Rot: ({rot_x}, {rot_y}) Zoom: {zoom}")
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
                last_mouse = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_down = False
        elif event.type == pygame.MOUSEMOTION and mouse_down:
            x, y = pygame.mouse.get_pos()
            rot_y += (x - last_mouse[0]) * 0.5
            rot_x += (y - last_mouse[1]) * 0.5
            last_mouse = (x, y)
    
    return True

# ==========================================
# 5. BUCLE PRINCIPAL
# ==========================================
print("\n" + "="*50)
print("VISUALIZADOR 3D INICIADO")
print("="*50)
print("Controles:")
print("  - Mouse arrastrar: Rotar vista")
print("  - Flechas: Rotar")
print("  - +/- : Zoom in/out")
print("  - ESC: Salir")
print("="*50)

clock = pygame.time.Clock()
corriendo = True

while corriendo:
    corriendo = manejar_eventos()
    dibujar()
    clock.tick(60)  # Limitar a 60 FPS

pygame.quit()
sys.exit()