# visualization/opengl_heladas.py
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.simulador_heladas import simular_heladas_desde_csv

# ================== CONFIGURACIÓN ==================
RUTA_CSV = 'datos_heladas_altiplano.csv'
ANCHO_VENTANA = 1200
ALTO_VENTANA = 800
EXAGERACION = 3.0
# ==================================================

# Colormap para heladas
COLORES_HELADA = np.array([
    [0.0, 0.0, 0.5],    # Azul oscuro - seguro
    [0.0, 0.5, 1.0],    # Azul claro
    [0.0, 0.8, 0.0],    # Verde
    [1.0, 1.0, 0.0],    # Amarillo
    [1.0, 0.5, 0.0],    # Naranja
    [1.0, 0.0, 0.0]     # Rojo - helada severa
])

def obtener_color(prob):
    """Convierte probabilidad a color RGB"""
    idx = prob * (len(COLORES_HELADA) - 1)
    i = int(idx)
    f = idx - i
    if i >= len(COLORES_HELADA) - 1:
        return COLORES_HELADA[-1]
    return COLORES_HELADA[i] + (COLORES_HELADA[i+1] - COLORES_HELADA[i]) * f

def crear_vertices(X, Y, Z, prob, exag=3.0):
    """Crea malla 3D con colores"""
    ny, nx = Z.shape
    vertices = []
    colores = []
    triangulos = []
    
    # Normalizar coordenadas
    x_norm = (X - X.min()) / (X.max() - X.min()) * 2 - 1
    y_norm = (Y - Y.min()) / (Y.max() - Y.min()) * 2 - 1
    z_norm = (Z - Z.min()) / (Z.max() - Z.min()) * exag
    
    # Crear vértices
    for i in range(ny):
        for j in range(nx):
            vertices.append([x_norm[i,j], z_norm[i,j], y_norm[i,j]])
            colores.append(obtener_color(prob[i,j]))
    
    # Crear triángulos
    for i in range(ny-1):
        for j in range(nx-1):
            v1 = i*nx + j
            v2 = v1 + 1
            v3 = (i+1)*nx + j
            v4 = v3 + 1
            triangulos.append([v1, v2, v3])
            triangulos.append([v2, v4, v3])
    
    return np.array(vertices), np.array(colores), np.array(triangulos)

def guardar_screenshot():
    """Guarda la pantalla actual como imagen PNG"""
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"outputs/helada_opengl_{timestamp}.png"
    pygame.image.save(pygame.display.get_surface(), nombre)
    print(f"\n   📸 Screenshot guardado: {nombre}")

def main():
    print("\n" + "="*55)
    print("🏔️  VISOR OPENGL - PREDICCIÓN DE HELADAS ALTIPLANO")
    print("="*55)
    
    # Generar datos desde CSV
    print("\n📊 Generando simulación de heladas...")
    X, Y, Z, probabilidad = simular_heladas_desde_csv(RUTA_CSV)
    
    # Crear malla 3D
    print("🔧 Creando malla 3D...")
    vertices, colores, triangulos = crear_vertices(X, Y, Z, probabilidad, EXAGERACION)
    print(f"   ✅ {len(vertices)} vértices, {len(triangulos)} triángulos")
    
    # Inicializar Pygame
    print("\n🎮 Iniciando OpenGL...")
    pygame.init()
    pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA), DOUBLEBUF|OPENGL)
    pygame.display.set_caption("🌄 MAPA 3D HELADAS - ALTIPLANO PERUANO | Mouse: Rotar | Scroll: Zoom | ESPACIO: Foto | ESC: Salir")
    
    # Configurar OpenGL
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.05, 0.12, 1.0)
    gluPerspective(45, ANCHO_VENTANA/ALTO_VENTANA, 0.1, 50.0)
    
    # Variables de cámara
    rot_x, rot_y = 35, 45
    zoom = 4.5
    running = True
    dragging = False
    
    print("\n✅ ¡VENTANA ABIERTA! Mira tu pantalla.")
    print("="*55)
    print("🎮 CONTROLES:")
    print("   🖱️  Click izquierdo + arrastrar = Rotar terreno")
    print("   🔍 Scroll = Zoom in/out")
    print("   📸 ESPACIO = Guardar captura para artículo")
    print("   🔄 R = Resetear vista")
    print("   ❌ ESC = Salir")
    print("="*55 + "\n")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("\n👋 Cerrando visor...")
                    running = False
                elif event.key == pygame.K_SPACE:
                    guardar_screenshot()
                elif event.key == pygame.K_r:
                    rot_x, rot_y = 35, 45
                    zoom = 4.5
                    print("   🔄 Vista reseteada")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    pygame.mouse.get_rel()
                elif event.button == 4:
                    zoom = max(1.5, zoom - 0.3)
                elif event.button == 5:
                    zoom = min(15.0, zoom + 0.3)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx, dy = pygame.mouse.get_rel()
                    rot_y += dx * 0.4
                    rot_x += dy * 0.4
        
        # Renderizar escena
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, -1.5, -zoom)
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        
        # Dibujar terreno con colores de helada
        glBegin(GL_TRIANGLES)
        for tri in triangulos:
            for vid in tri:
                glColor3fv(colores[vid])
                glVertex3fv(vertices[vid])
        glEnd()
        
        # Wireframe suave para ver relieve
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glColor3f(0.15, 0.15, 0.15)
        glLineWidth(0.3)
        glBegin(GL_TRIANGLES)
        for tri in triangulos:
            for vid in tri:
                glVertex3fv(vertices[vid])
        glEnd()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        pygame.display.flip()
        pygame.time.wait(10)
    
    pygame.quit()
    print("✅ Programa finalizado correctamente.\n")

if __name__ == "__main__":
    main()