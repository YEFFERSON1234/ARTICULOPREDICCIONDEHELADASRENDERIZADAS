"""
Terrain_mesh.py
Convierte archivos DEM (Digital Elevation Model) a vértices normalizados para OpenGL
"""

import numpy as np
import pandas as pd
import gzip
import sys

# Configurar encoding para Windows (solo si no está ya configurado)
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class TerrainMesh:
    def __init__(self, dem_csv_path='Render/Mapa/dem_puno_render.csv.gz'):
        """
        Inicializa la malla del terreno desde un archivo CSV comprimido
        
        Args:
            dem_csv_path: Ruta al archivo CSV comprimido con datos DEM
        """
        print(f"[TerrainMesh] Cargando DEM desde {dem_csv_path}...")
        
        # Cargar datos DEM
        if dem_csv_path.endswith('.gz'):
            df = pd.read_csv(dem_csv_path, compression='gzip')
        else:
            df = pd.read_csv(dem_csv_path)
        
        print(f"[TerrainMesh] Registros DEM cargados: {len(df)}")
        
        # Extraer coordenadas y elevación
        self.longitude = df['longitud'].values
        self.latitude = df['latitud'].values
        self.elevation = df['elevacion'].values
        
        # Normalizar coordenadas a rango [0, 1] para OpenGL
        self.vertices = self._normalize_vertices()
        
        # Calcular normales para iluminación
        self.normals = self._calculate_normals()
        
        print(f"[TerrainMesh] Malla del terreno generada: {len(self.vertices)} vértices")
    
    def _normalize_vertices(self):
        """
        Normaliza las coordenadas del terreno a rango [0, 1] para OpenGL
        
        Returns:
            numpy array: vértices normalizados [x, y, z]
        """
        # Normalizar longitud a x [0, 1]
        x = (self.longitude - self.longitude.min()) / (self.longitude.max() - self.longitude.min())
        
        # Normalizar latitud a y [0, 1]
        y = (self.latitude - self.latitude.min()) / (self.latitude.max() - self.latitude.min())
        
        # Normalizar elevación a z [0, 1] (escala vertical)
        z = (self.elevation - self.elevation.min()) / (self.elevation.max() - self.elevation.min())
        
        # Escalar z para que el terreno tenga más relieve visual
        z = z * 0.3  # Factor de escala vertical
        
        # Crear array de vértices [x, y, z]
        vertices = np.column_stack([x, y, z])
        
        return vertices
    
    def _calculate_normals(self):
        """
        Calcula los vectores normales para cada vértice para iluminación OpenGL
        
        Returns:
            numpy array: normales [nx, ny, nz]
        """
        # Para simplificar, usamos normales hacia arriba (0, 0, 1)
        # En una implementación más completa, se calcularían usando gradientes
        normals = np.zeros_like(self.vertices)
        normals[:, 2] = 1.0  # Normal hacia arriba (eje Z)
        
        return normals
    
    def get_vertices(self):
        """Retorna los vértices normalizados"""
        return self.vertices
    
    def get_normals(self):
        """Retorna los normales"""
        return self.normals
    
    def get_original_coordinates(self):
        """Retorna las coordenadas originales (lat, lon, elev)"""
        return {
            'longitude': self.longitude,
            'latitude': self.latitude,
            'elevation': self.elevation
        }
    
    def save_to_file(self, output_path='../Render/Mapa/terrain_vertices.npy'):
        """
        Guarda los vértices normalizados en un archivo .npy para carga rápida
        
        Args:
            output_path: Ruta de salida
        """
        np.save(output_path, self.vertices)
        print(f"[TerrainMesh] Vértices guardados en {output_path}")


def main():
    """Función principal para probar la generación de malla"""
    print("="*70)
    print("TERRAIN MESH GENERATOR")
    print("="*70)
    
    # Crear malla del terreno
    terrain = TerrainMesh()
    
    # Mostrar estadísticas
    print(f"\nEstadísticas de vértices:")
    print(f"  Min X: {terrain.vertices[:, 0].min():.4f}, Max X: {terrain.vertices[:, 0].max():.4f}")
    print(f"  Min Y: {terrain.vertices[:, 1].min():.4f}, Max Y: {terrain.vertices[:, 1].max():.4f}")
    print(f"  Min Z: {terrain.vertices[:, 2].min():.4f}, Max Z: {terrain.vertices[:, 2].max():.4f}")
    
    # Guardar vértices
    terrain.save_to_file()
    
    print(f"\n[OK] Malla del terreno generada exitosamente")


if __name__ == '__main__':
    main()
