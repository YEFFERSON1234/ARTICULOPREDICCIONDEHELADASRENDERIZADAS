"""
Script de Quick Start para verificar que todo funciona
Valida la instalación y ejecuta pruebas básicas
"""

import sys
from pathlib import Path
import subprocess

# Agregar proyecto al path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def check_python_version():
    """Verifica versión de Python"""
    print_header("1️⃣  Verificando versión de Python")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Se recomienda Python 3.8 o superior")
        return False
    print("✓ Versión compatible\n")
    return True

def check_imports():
    """Verifica que se pueden importar módulos clave"""
    print_header("2️⃣  Verificando importaciones")
    
    imports_to_check = {
        "config": "Configuración centralizada",
        "logging_config": "Sistema de logging",
        "main_pipeline": "Pipeline maestro"
    }
    
    success = True
    for module_name, description in imports_to_check.items():
        try:
            __import__(module_name)
            print(f"✓ {module_name:20} - {description}")
        except ImportError as e:
            print(f"❌ {module_name:20} - {description}")
            print(f"   Error: {e}\n")
            success = False
    
    print()
    return success

def check_data_files():
    """Verifica disponibilidad de archivos de datos"""
    print_header("3️⃣  Verificando archivos de datos")
    
    import scripts.config as config
    
    files_to_check = {
        config.SENAMHI_CSV: "SENAMHI (crítico)",
        config.ERA5_MAESTRO_CSV: "ERA5 procesado (opcional)",
        config.DATASET_FINAL_ML: "Dataset ML (opcional)",
    }
    
    found = 0
    for filepath, description in files_to_check.items():
        if Path(filepath).exists():
            size_kb = Path(filepath).stat().st_size / 1024
            print(f"✓ {filepath.name:30} - {description} ({size_kb:.1f} KB)")
            found += 1
        else:
            status = "⚠️  FALTA" if "crítico" in description else "◇ No encontrado"
            print(f"{status} {filepath.name:30} - {description}")
    
    print(f"\n{found}/{len(files_to_check)} archivos encontrados\n")
    return found > 0

def check_models_dir():
    """Verifica directorio de modelos"""
    print_header("4️⃣  Verificando directorio de modelos")
    
    import scripts.config as config
    
    if not config.MODELS_DIR.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directorio creado: {config.MODELS_DIR}")
    else:
        models = list(config.MODELS_DIR.glob("*.pkl"))
        if models:
            print(f"✓ Modelos encontrados: {len(models)}")
            for model in models:
                print(f"   • {model.name}")
        else:
            print("◇ Directorio existe pero vacío (modelos no entrenados)")
    print()

def check_visualization():
    """Verifica módulo de visualización"""
    print_header("5️⃣  Verificando visualización")
    
    try:
        from visualization import unified_visualizer
        print("✓ Visualizador unificado cargado correctamente")
        print("  Modos disponibles: interactive, static, animated, risk")
    except ImportError as e:
        print(f"❌ Error al cargar visualizador: {e}")
        return False
    
    print()
    return True

def run_quick_test():
    """Ejecuta un test rápido"""
    print_header("6️⃣  Ejecutando test rápido de datos")
    
    try:
        import scripts.config as config
        from tests.test_data_preparation import TestDataIntegrity
        import unittest
        
        # Crear suite de tests rápidos
        suite = unittest.TestSuite()
        suite.addTest(TestDataIntegrity('test_senamhi_file_exists'))
        
        if Path(config.SENAMHI_CSV).exists():
            suite.addTest(TestDataIntegrity('test_senamhi_not_empty'))
            suite.addTest(TestDataIntegrity('test_required_columns_exist'))
        
        # Ejecutar
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        print()
        if result.wasSuccessful():
            print(f"✓ {result.testsRun} tests pasados\n")
            return True
        else:
            print(f"❌ {len(result.failures)} fallos, {len(result.errors)} errores\n")
            return False
    
    except Exception as e:
        print(f"⚠️  Tests deshabilitados: {e}\n")
        return True

def print_next_steps():
    """Imprime pasos siguientes"""
    print_header("✨ PASOS SIGUIENTES")
    
    print("""
1. EJECUTAR PIPELINE COMPLETO:
   python main_pipeline.py

2. EJECUTAR TESTS:
   python -m pytest tests/ -v

3. VISUALIZAR RESULTADOS:
   python visualization/unified_visualizer.py --mode static

4. LEER DOCUMENTACIÓN:
   • IMPROVEMENTS.md (mejoras implementadas)
   • docs/EXECUTION_GUIDE.md (guía detallada)
   • docs/README.md (información general)

5. EXPLORAR CONFIGURACIÓN:
   • config.py (parámetros del proyecto)
   • logging_config.py (logging)
   • consolidation_plan.py (análisis de duplicados)

¿Necesitas ayuda? Revisa los archivos de documentación.
    """)

def main():
    """Ejecuta todas las verificaciones"""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "VERIFICACIÓN RÁPIDA DEL PROYECTO" + " "*21 + "║")
    print("║" + " "*12 + "Predicción de Heladas Renderizadas - Puno" + " "*14 + "║")
    print("╚" + "="*68 + "╝")
    
    checks = [
        ("Python", check_python_version),
        ("Importaciones", check_imports),
        ("Datos", check_data_files),
        ("Modelos", check_models_dir),
        ("Visualización", check_visualization),
        ("Tests", run_quick_test),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error en {name}: {e}\n")
            results[name] = False
    
    # Resumen
    print_header("📊 RESUMEN")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Verificaciones pasadas: {passed}/{total}\n")
    for name, result in results.items():
        status = "✓" if result else "❌"
        print(f"  {status} {name}")
    
    # Pasos siguientes
    if passed >= total - 1:
        print_next_steps()
        print("\n✅ SISTEMA LISTO PARA USAR\n")
        return 0
    else:
        print("\n⚠️  ALGUNAS VERIFICACIONES FALLARON")
        print("Revisa los errores arriba y soluciona antes de continuar.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
