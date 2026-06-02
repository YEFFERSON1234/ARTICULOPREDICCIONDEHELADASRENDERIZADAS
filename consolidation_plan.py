"""
Script de utilidad para consolidar y refactorizar el código duplicado
"""

import os
from pathlib import Path

def analyze_duplicates():
    """Analiza archivos potencialmente duplicados"""
    
    duplicates = {
        "SVM": ["modelos/SVM.py", "modelos/SVM_senamhi.py"],
        "Visualizadores": [
            "Render/Mapa/draw_from_csv.py",
            "Render/Mapa/visualizer_with_risk.py",
            "Render/Mapa/animated_visualizer.py",
            "visualization/main.py"
        ],
        "Procesamiento ERA5": [
            "utils/process_csv.py",
            "modelos/unificar_datos.ipynb"
        ]
    }
    
    print("\n" + "="*70)
    print("ANÁLISIS DE CÓDIGO DUPLICADO")
    print("="*70 + "\n")
    
    for category, files in duplicates.items():
        print(f"\n📋 {category}:")
        for file in files:
            exists = Path(file).exists()
            status = "✓ Existe" if exists else "❌ No existe"
            size = ""
            if exists:
                size = f" ({Path(file).stat().st_size / 1024:.1f} KB)"
            print(f"   • {file}{size} {status}")
    
    print("\n" + "="*70)
    print("RECOMENDACIONES:")
    print("="*70)
    print("""
1. CONSOLIDAR SVM:
   • Crear: modelos/svm_model.py (unificado)
   • Referencia: Revisar ambos archivos y crear versión única
   
2. CONSOLIDAR VISUALIZADORES:
   • Ya se hizo: visualization/unified_visualizer.py
   • Acción: Los otros scripts ahora son opcionales (legacy)
   
3. CONSOLIDAR ERA5:
   • Crear: utils/era5_processor.py (importable)
   • Mover lógica de .ipynb a .py
   • Referencia: data_process/unify_data.py

4. LIMPIEZA:
   • Después de consolidar, mover archivos legacy a carpeta "deprecated/"
   • Mantener referencias en DEPRECATION.md
    """)


def generate_consolidation_checklist():
    """Genera un checklist para consolidación"""
    
    checklist = """
CHECKLIST DE CONSOLIDACIÓN DE CÓDIGO
=====================================

[ ] 1. SVM - Consolidation
    [ ] 1.1 Leer SVM.py
    [ ] 1.2 Leer SVM_senamhi.py
    [ ] 1.3 Identificar diferencias
    [ ] 1.4 Crear svm_unified.py
    [ ] 1.5 Probar con datos de ejemplo
    [ ] 1.6 Actualizar train.py para usar versión unificada

[ ] 2. Visualizadores - Consolidation
    [ ] 2.1 Crear alias en main.py hacia unified_visualizer.py
    [ ] 2.2 Documentar modo legacy vs nuevo
    [ ] 2.3 Crear deprecation warnings en archivos antiguos

[ ] 3. ERA5 - Consolidation
    [ ] 3.1 Extraer lógica de unificar_datos.ipynb
    [ ] 3.2 Crear utils/era5_processor.py
    [ ] 3.3 Hacer importable desde main_pipeline.py
    [ ] 3.4 Agregar tests para ERA5

[ ] 4. Testing - Expansion
    [ ] 4.1 Crear tests/test_models.py
    [ ] 4.2 Crear tests/test_preprocessing.py
    [ ] 4.3 Crear tests/test_integration.py
    [ ] 4.4 Configurar CI/CD (.github/workflows/)

[ ] 5. Documentación
    [ ] 5.1 Actualizar README.md
    [ ] 5.2 Crear ARCHITECTURE.md
    [ ] 5.3 Crear CONTRIBUTING.md
    [ ] 5.4 Crear Jupyter notebooks de ejemplo

[ ] 6. Git/Versionado
    [ ] 6.1 Crear rama para refactoring
    [ ] 6.2 Hacer commits pequeños y descriptivos
    [ ] 6.3 Ejecutar tests antes de cada commit
    [ ] 6.4 Crear Pull Request con descripción detallada
"""
    
    return checklist


if __name__ == "__main__":
    print("\n🔍 Analizando estructura del proyecto...\n")
    analyze_duplicates()
    
    print("\n" + generate_consolidation_checklist())
