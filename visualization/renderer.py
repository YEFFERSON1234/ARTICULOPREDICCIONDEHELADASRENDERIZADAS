"""
Renderer.py
Motor de renderizado 3D con OpenGL para visualización del terreno y predicciones de heladas
"""

import numpy as np
import sys
from .terrain_mesh import TerrainMesh

# Configurar encoding para Windows (solo si no está ya configurado)
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import pygame
    from pygame.locals import *
    OPENGL_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] OpenGL/Pygame no disponible: {e}")
    print("Instala con: pip install PyOpenGL pygame")
    OPENGL_AVAILABLE = False


class Renderer:
    def __init__(self, width=1024, height=768, title="Predicción de Heladas - Puno 3D"):
        """
        Inicializa el motor de renderizado OpenGL
        
        Args:
            width: Ancho de la ventana
            height: Altura de la ventana
            title: Título de la ventana
        """
        if not OPENGL_AVAILABLE:
            raise RuntimeError("OpenGL/Pygame no está instalado")
        
        self.width = width
        self.height = height
        self.title = title
        
        # Variables de cámara
        self.rotation_x = 45.0  # Rotación en X (elevación)
        self.rotation_y = -45.0  # Rotación en Y (azimut)
        self.zoom = 3.0          # Distancia de cámara
        self.pan_x = 0.0         # Desplazamiento X
        self.pan_y = 0.0         # Desplazamiento Y
        
        # Variables de animación
        self.auto_rotate = False
        self.rotation_speed = 0.5
        
        # Datos del terreno
        self.terrain = None
        self.vertices = None
        self.frost_data = None
        
        # Estado del mouse
        self.mouse_down = False
        self.last_mouse_pos = (0, 0)
        
        print("[Renderer] Motor OpenGL inicializado")
    
    def init_opengl(self):
        """Inicializa el contexto OpenGL"""
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption(self.title)
        
        # Configurar proyección
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 100.0)
        
        # Configurar modelo
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Habilitar profundidad
        glEnable(GL_DEPTH_TEST)
        
        # Habilitar iluminación básica
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glEnable(GL_COLOR_MATERIAL)
        
        print("[Renderer] OpenGL inicializado correctamente")
    
    def load_terrain(self, terrain_path='../Render/Mapa/terrain_vertices.npy'):
        """
        Carga los vértices del terreno
        
        Args:
            terrain_path: Ruta al archivo .npy con vértices
        """
        try:
            self.vertices = np.load(terrain_path)
            print(f"[Renderer] Terreno cargado: {len(self.vertices)} vértices")
        except FileNotFoundError:
            # Si no existe el archivo .npy, generar desde TerrainMesh
            print(f"[Renderer] Generando terreno desde DEM...")
            from .terrain_mesh import TerrainMesh
            terrain = TerrainMesh()
            self.vertices = terrain.get_vertices()
            terrain.save_to_file(terrain_path)
    
    def load_frost_predictions(self, predictions_path='../data_process/predictions.csv'):
        """
        Carga las predicciones de heladas para el mapa de calor
        
        Args:
            predictions_path: Ruta al archivo predictions.csv
        """
        try:
            import pandas as pd
            df = pd.read_csv(predictions_path)
            self.frost_data = df
            print(f"[Renderer] Predicciones cargadas: {len(df)} registros")
        except FileNotFoundError:
            print(f"[WARNING] Archivo de predicciones no encontrado: {predictions_path}")
            print("[WARNING] El renderizado continuará sin datos de heladas")
        except Exception as e:
            print(f"[WARNING] No se pudieron cargar predicciones: {e}")
            print("[WARNING] El renderizado continuará sin datos de heladas")
    
    def get_frost_color(self, prob_helada):
        """
        Convierte probabilidad de helada a color RGB
        
        Args:
            prob_helada: Probabilidad de helada (0-1)
        
        Returns:
            tuple: Color RGB (0-1)
        """
        # Verde (sin riesgo) -> Amarillo -> Rojo (alto riesgo)
        if prob_helada < 0.3:
            # Verde a amarillo
            r = prob_helada / 0.3
            g = 1.0
            b = 0.0
        elif prob_helada < 0.7:
            # Amarillo a rojo
            r = 1.0
            g = 1.0 - (prob_helada - 0.3) / 0.4
            b = 0.0
        else:
            # Rojo intenso
            r = 1.0
            g = 0.0
            b = (prob_helada - 0.7) / 0.3 * 0.5
        
        return (r, g, b)
    
    def render_terrain(self):
        """Renderiza el terreno 3D"""
        if self.vertices is None:
            return
        
        glBegin(GL_POINTS)
        for vertex in self.vertices:
            # Color basado en elevación (azul bajo, verde medio, blanco alto)
            elevation = vertex[2] / 0.3  # Normalizar a 0-1
            if elevation < 0.3:
                glColor3f(0.0, 0.0, 0.5 + elevation)  # Azul
            elif elevation < 0.7:
                glColor3f(0.0, 1.0, 0.0)  # Verde
            else:
                glColor3f(1.0, 1.0, 1.0)  # Blanco (nieve)
            
            glVertex3f(vertex[0], vertex[1], vertex[2])
        glEnd()
    
    def render_frost_heatmap(self):
        """Renderiza el mapa de calor de heladas sobre el terreno"""
        if self.frost_data is None or self.vertices is None:
            return
        
        # Para simplificar, renderizamos puntos de predicción
        # En una implementación completa, se interpolarían sobre el terreno
        glBegin(GL_POINTS)
        for _, row in self.frost_data.iterrows():
            # Normalizar lat/lon a coordenadas de pantalla
            lat_norm = (row['lat'] - (-17.5)) / ( -14.5 - (-17.5))
            lon_norm = (row['lon'] - (-72.0)) / (-68.0 - (-72.0))
            
            # Obtener color basado en probabilidad de helada
            color = self.get_frost_color(row['prob_helada'])
            glColor3f(*color)
            
            # Renderizar punto elevado sobre el terreno
            glVertex3f(lon_norm, lat_norm, 0.35 + row['prob_helada'] * 0.1)
        glEnd()
    
    def handle_input(self):
        """Maneja la entrada del usuario"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == K_PLUS or event.key == K_EQUALS:
                    self.zoom = max(1.0, self.zoom - 0.2)
                elif event.key == K_MINUS:
                    self.zoom = min(10.0, self.zoom + 0.2)
                elif event.key == K_SPACE:
                    self.auto_rotate = not self.auto_rotate
                elif event.key == K_r:
                    # Reset cámara
                    self.rotation_x = 45.0
                    self.rotation_y = -45.0
                    self.zoom = 3.0
                    self.pan_x = 0.0
                    self.pan_y = 0.0
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Click izquierdo
                    self.mouse_down = True
                    self.last_mouse_pos = event.pos
                elif event.button == 4:  # Scroll up
                    self.zoom = max(1.0, self.zoom - 0.2)
                elif event.button == 5:  # Scroll down
                    self.zoom = min(10.0, self.zoom + 0.2)
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
            
            elif event.type == MOUSEMOTION:
                if self.mouse_down:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.rotation_y += dx * 0.5
                    self.rotation_x += dy * 0.5
                    self.rotation_x = max(-90, min(90, self.rotation_x))
                    self.last_mouse_pos = event.pos
        
        # Controles de teclado continuos
        keys = pygame.key.get_pressed()
        if keys[K_LEFT]:
            self.rotation_y -= 1.0
        if keys[K_RIGHT]:
            self.rotation_y += 1.0
        if keys[K_UP]:
            self.rotation_x -= 1.0
            self.rotation_x = max(-90, self.rotation_x)
        if keys[K_DOWN]:
            self.rotation_x += 1.0
            self.rotation_x = min(90, self.rotation_x)
        
        return True
    
    def update_camera(self):
        """Actualiza la posición de la cámara"""
        glLoadIdentity()
        
        # Aplicar zoom
        glTranslatef(0.0, 0.0, -self.zoom)
        
        # Aplicar rotación
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        
        # Aplicar pan
        glTranslatef(self.pan_x, self.pan_y, 0.0)
        
        # Centrar el terreno
        glTranslatef(-0.5, -0.5, 0.0)
    
    def render(self):
        """Renderiza un frame"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.update_camera()
        
        # Renderizar terreno
        self.render_terrain()
        
        # Renderizar mapa de calor de heladas
        self.render_frost_heatmap()
        
        pygame.display.flip()
    
    def run(self):
        """Ejecuta el bucle principal de renderizado"""
        print("[Renderer] Iniciando bucle de renderizado...")
        print("[Renderer] Controles:")
        print("  - Mouse arrastrar: rotar")
        print("  - Scroll / +/-: zoom")
        print("  - Flechas: rotar cámara")
        print("  - Espacio: auto-rotación")
        print("  - R: reset cámara")
        print("  - ESC: salir")
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            running = self.handle_input()
            
            # Auto-rotación
            if self.auto_rotate:
                self.rotation_y += self.rotation_speed
            
            self.render()
            clock.tick(60)  # 60 FPS
        
        pygame.quit()
        print("[Renderer] Renderizado finalizado")


def main():
    """Función principal para probar el renderer"""
    print("="*70)
    print("RENDERER 3D - PREDICCIÓN DE HELADAS")
    print("="*70)
    
    try:
        renderer = Renderer()
        renderer.init_opengl()
        renderer.load_terrain()
        renderer.load_frost_predictions()
        renderer.run()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
