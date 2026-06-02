"""
Main.py
Integración final: carga CSV de predicciones y renderiza en el motor OpenGL
Sistema de Visualización Tridimensional Interactiva con OpenGL para Predicción de Heladas
"""

import sys
import os

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("SISTEMA DE VISUALIZACIÓN 3D - PREDICCIÓN DE HELADAS")
print("Integración: ML + OpenGL")
print("="*70)

# Verificar archivos necesarios
print("\n[1/4] Verificando archivos necesarios...")

required_files = [
    '../data_process/predictions.csv',
    '../Render/Mapa/dem_puno_render.csv.gz',
    'Terrain_mesh.py',
    'Renderer.py'
]

missing_files = []
for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)
        print(f"  [X] Falta: {file}")
    else:
        print(f"  [OK] Existe: {file}")

if missing_files:
    print(f"\n[ERROR] Faltan archivos necesarios: {missing_files}")
    print("Por favor, ejecuta primero los scripts de preparación:")
    print("  1. python ../modelos/xgboost_model.py")
    print("  2. python ../data_process/generar_predictions_final.py")
    print("  3. python Terrain_mesh.py")
    sys.exit(1)

# Cargar predicciones
print("\n[2/4] Cargando predicciones de heladas...")
try:
    import pandas as pd
    predictions = pd.read_csv('../data_process/predictions.csv')
    print(f"  [OK] {len(predictions)} predicciones cargadas")
    print(f"  Columnas: {list(predictions.columns)}")
    print(f"  Rango de prob_helada: {predictions['prob_helada'].min():.3f} - {predictions['prob_helada'].max():.3f}")
except Exception as e:
    print(f"  [ERROR] No se pudieron cargar predicciones: {e}")
    sys.exit(1)

# Generar malla del terreno
print("\n[3/4] Generando malla del terreno...")
try:
    from .terrain_mesh import TerrainMesh
    terrain = TerrainMesh()
    vertices = terrain.get_vertices()
    print(f"  [OK] Malla generada: {len(vertices)} vértices")
except Exception as e:
    print(f"  [ERROR] No se pudo generar malla: {e}")
    sys.exit(1)

# Iniciar renderizado
print("\n[4/4] Iniciando motor de renderizado OpenGL...")
try:
    from .renderer import Renderer
    renderer = Renderer()
    renderer.init_opengl()
    renderer.load_terrain()
    renderer.load_frost_predictions()
    renderer.run()
except Exception as e:
    print(f"  [ERROR] No se pudo iniciar renderer: {e}")
    print("\nNota: OpenGL/Pygame deben estar instalados:")
    print("  pip install PyOpenGL pygame")
    sys.exit(1)

print("\n[OK] Visualización finalizada")
