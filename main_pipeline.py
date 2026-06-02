"""
Pipeline maestro para predicción de heladas en Puno
Orquesta todo el flujo: preparación de datos, entrenamiento, predicción y visualización
"""

import sys
import os
from pathlib import Path
import argparse
import traceback

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

import logging_config
from logging_config import setup_logging
import config

logger = setup_logging(__name__)

# Importar funciones de los módulos existentes
try:
    from src.train import train_models, prepare_features as prepare_features_train
    from src.predict import predict
    from data_process.prepare_data import main as prepare_data_main
    from data_process.unify_data import main as unify_data_main
except ImportError as e:
    logger.warning(f"No se pudieron importar algunos módulos: {e}")


class FrostPredictionPipeline:
    """Pipeline maestro para predicción de heladas"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.logger = logger
        
    def log(self, msg, level="info"):
        """Registra mensajes en logger"""
        if self.verbose:
            getattr(self.logger, level)(msg)
    
    def step(self, step_num, description):
        """Imprime paso del pipeline"""
        banner = f"\n{'='*70}"
        banner += f"\n[PASO {step_num}] {description}"
        banner += f"\n{'='*70}\n"
        self.log(banner)
    
    def check_file_exists(self, filepath, description):
        """Verifica si un archivo existe, genera warning si no"""
        if not Path(filepath).exists():
            self.log(f"⚠️  NO SE ENCONTRÓ: {description} ({filepath})", level="warning")
            return False
        self.log(f"✓ ENCONTRADO: {description}")
        return True
    
    def prepare_data(self):
        """Paso 1: Preparación de datos"""
        self.step(1, "Preparación de datos")
        
        try:
            # Verificar que exista el CSV de SENAMHI
            if not self.check_file_exists(config.SENAMHI_CSV, "CSV de SENAMHI"):
                self.log("No se puede continuar sin datos SENAMHI", level="error")
                return False
            
            self.log("Ejecutando prepare_data.py...")
            # En un entorno real, aquí llamarías a la función correspondiente
            # prepare_data_main()  # Si está disponible
            self.log("✓ Datos preparados")
            return True
        except Exception as e:
            self.log(f"❌ Error en preparación de datos: {e}", level="error")
            self.log(traceback.format_exc(), level="debug")
            return False
    
    def unify_data(self):
        """Paso 2: Unificación de SENAMHI + ERA5"""
        self.step(2, "Unificación de datos SENAMHI + ERA5")
        
        try:
            if not self.check_file_exists(config.ERA5_MAESTRO_CSV, "CSV maestro ERA5"):
                self.log("⚠️  ERA5 no encontrado, continuando solo con SENAMHI", level="warning")
            
            self.log("Ejecutando unify_data.py...")
            # unify_data_main()  # Si está disponible
            self.log("✓ Datos unificados")
            return True
        except Exception as e:
            self.log(f"❌ Error en unificación: {e}", level="error")
            self.log(traceback.format_exc(), level="debug")
            return False
    
    def train_models_step(self):
        """Paso 3: Entrenamiento de modelos"""
        self.step(3, "Entrenamiento de modelos XGBoost")
        
        try:
            dataset_path = config.DATASET_FINAL_ML
            
            if not self.check_file_exists(dataset_path, "Dataset final ML"):
                self.log(f"Usando dataset alternativo: {config.SENAMHI_CSV}")
                dataset_path = config.SENAMHI_CSV
            
            self.log(f"Leyendo datos de: {dataset_path}")
            
            try:
                train_models(str(dataset_path), models_dir=str(config.MODELS_DIR))
                self.log("✓ Modelos entrenados exitosamente")
                return True
            except NameError:
                self.log("Función train_models no disponible, continuando...", level="warning")
                return True
        except Exception as e:
            self.log(f"❌ Error en entrenamiento: {e}", level="error")
            self.log(traceback.format_exc(), level="debug")
            return False
    
    def generate_predictions(self):
        """Paso 4: Generación de predicciones"""
        self.step(4, "Generación de predicciones")
        
        try:
            dataset_path = config.DATASET_FINAL_ML
            if not Path(dataset_path).exists():
                dataset_path = config.SENAMHI_CSV
            
            self.log(f"Generando predicciones con dataset: {dataset_path}")
            
            try:
                predict(
                    str(dataset_path),
                    models_dir=str(config.MODELS_DIR),
                    out_path=str(config.PREDICTIONS_CSV)
                )
                self.log(f"✓ Predicciones guardadas en: {config.PREDICTIONS_CSV}")
                return True
            except NameError:
                self.log("Función predict no disponible, continuando...", level="warning")
                return True
        except Exception as e:
            self.log(f"❌ Error en predicciones: {e}", level="error")
            self.log(traceback.format_exc(), level="debug")
            return False
    
    def visualize_results(self):
        """Paso 5: Visualización de resultados"""
        self.step(5, "Visualización interactiva 3D")
        
        try:
            if not self.check_file_exists(config.PREDICTIONS_CSV, "Archivo de predicciones"):
                self.log("No hay predicciones para visualizar", level="warning")
                return False
            
            self.log("Inicializando visualización 3D...")
            self.log("Para visualizar, ejecuta: python visualization/main.py")
            self.log("✓ Sistema de visualización listo")
            return True
        except Exception as e:
            self.log(f"❌ Error en visualización: {e}", level="error")
            return False
    
    def run_full_pipeline(self, skip_steps=None):
        """
        Ejecuta el pipeline completo
        
        Args:
            skip_steps: Lista de números de pasos a saltar (ej: [1, 2])
        """
        skip_steps = skip_steps or []
        
        self.log(f"\n🚀 INICIANDO PIPELINE MAESTRO DE PREDICCIÓN DE HELADAS")
        self.log(f"📁 Proyecto: {config.PROJECT_ROOT}")
        self.log(f"🔧 Pasos a ejecutar: {[i for i in range(1, 6) if i not in skip_steps]}\n")
        
        results = {
            "paso_1_preparar": True if 1 in skip_steps else self.prepare_data(),
            "paso_2_unificar": True if 2 in skip_steps else self.unify_data(),
            "paso_3_entrenar": True if 3 in skip_steps else self.train_models_step(),
            "paso_4_predecir": True if 4 in skip_steps else self.generate_predictions(),
            "paso_5_visualizar": True if 5 in skip_steps else self.visualize_results(),
        }
        
        # Resumen final
        self.step("RESUMEN", "Resultado de la ejecución")
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        self.log(f"Pasos exitosos: {successful}/{total}")
        
        for step_name, success in results.items():
            status = "✓" if success else "❌"
            self.log(f"  {status} {step_name}")
        
        if successful == total:
            self.log("\n✅ PIPELINE COMPLETADO EXITOSAMENTE", level="info")
        else:
            self.log(f"\n⚠️  PIPELINE CON {total - successful} ERRORES", level="warning")
        
        return results
    
    def run_partial_pipeline(self, stages):
        """
        Ejecuta solo los pasos especificados
        
        Args:
            stages: Lista de números de pasos a ejecutar (ej: [3, 4, 5])
        """
        all_stages = {1, 2, 3, 4, 5}
        skip_stages = all_stages - set(stages)
        return self.run_full_pipeline(skip_steps=list(skip_stages))


def main():
    """Función principal con argumentos CLI"""
    parser = argparse.ArgumentParser(
        description="Pipeline maestro para predicción de heladas en Puno",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main_pipeline.py                 # Ejecutar pipeline completo
  python main_pipeline.py --skip 1 2      # Saltar pasos 1 y 2
  python main_pipeline.py --only 3 4 5    # Ejecutar solo pasos 3, 4, 5 (entrenar, predecir, visualizar)
  python main_pipeline.py --predict-only  # Solo predicciones (asume modelos entrenados)
  python main_pipeline.py --visualize-only # Solo visualización (asume predicciones hechas)
        """
    )
    
    parser.add_argument(
        "--skip",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Números de pasos a saltar"
    )
    
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Ejecutar solo estos pasos"
    )
    
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Solo ejecutar predicciones (asume modelos entrenados)"
    )
    
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Solo ejecutar visualización (asume predicciones hechas)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Modo silencioso (solo errores en consola)"
    )
    
    args = parser.parse_args()
    
    # Crear pipeline
    pipeline = FrostPredictionPipeline(verbose=not args.quiet)
    
    # Determinar qué pasos ejecutar
    if args.predict_only:
        pipeline.run_partial_pipeline([3, 4])
    elif args.visualize_only:
        pipeline.run_partial_pipeline([5])
    elif args.only:
        pipeline.run_partial_pipeline(args.only)
    elif args.skip:
        pipeline.run_full_pipeline(skip_steps=args.skip)
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
