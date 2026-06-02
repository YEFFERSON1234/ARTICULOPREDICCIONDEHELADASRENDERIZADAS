"""
Visualizador Unificado para Predicción de Heladas en Puno
Consolida múltiples modos de visualización: estático, interactivo, animado
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Agregar rutas al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from logging_config import setup_logging

logger = setup_logging(__name__)

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class UnifiedVisualizer:
    """Visualizador unificado para predicciones de heladas"""
    
    def __init__(self, mode="interactive"):
        """
        Inicializa el visualizador
        
        Args:
            mode: Tipo de visualización
                - "interactive": 3D interactivo con OpenGL (por defecto)
                - "static": Mapa estático 2D
                - "animated": Animación temporal
                - "risk": Mapa de riesgo sobrepuesto
        """
        self.mode = mode
        self.logger = logger
        self.predictions_file = config.PREDICTIONS_CSV
        self.dem_file = config.RENDER_DIR / "dem_puno_render.csv.gz"
        
    def validate_requirements(self):
        """Valida que existan los archivos necesarios"""
        self.logger.info(f"[VALIDACIÓN] Verificando archivos para modo: {self.mode}")
        
        missing = []
        
        if not Path(self.predictions_file).exists():
            missing.append(f"Predicciones: {self.predictions_file}")
        
        if not Path(self.dem_file).exists():
            missing.append(f"DEM: {self.dem_file}")
        
        if missing:
            self.logger.error("❌ Archivos faltantes:")
            for file in missing:
                self.logger.error(f"   - {file}")
            return False
        
        self.logger.info("✓ Todos los archivos requeridos encontrados")
        return True
    
    def run_interactive(self):
        """Modo: Visualización 3D interactiva con OpenGL"""
        self.logger.info("\n" + "="*70)
        self.logger.info("MODO: VISUALIZACIÓN 3D INTERACTIVA (OpenGL)")
        self.logger.info("="*70)
        
        try:
            # Intenta importar y ejecutar el visualizador principal
            from renderer import TerrainRenderer
            from terrain_mesh import TerrainMesh
            
            self.logger.info("Cargando malla de terreno...")
            mesh = TerrainMesh(str(self.dem_file))
            
            self.logger.info("Inicializando renderizador OpenGL...")
            renderer = TerrainRenderer(mesh)
            
            self.logger.info("Cargando predicciones de heladas...")
            renderer.load_predictions(str(self.predictions_file))
            
            self.logger.info("\n" + "="*70)
            self.logger.info("CONTROLES:")
            self.logger.info("  • Mouse drag: Rotar vista")
            self.logger.info("  • Scroll / +/-: Zoom")
            self.logger.info("  • Flechas: Mover cámara")
            self.logger.info("  • ESC: Salir")
            self.logger.info("="*70 + "\n")
            
            renderer.run()
            
        except ImportError as e:
            self.logger.error(f"❌ No se pudieron importar módulos OpenGL: {e}")
            self.logger.info("💡 Intenta modo --static o --animated")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error en visualización 3D: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
        
        return True
    
    def run_static(self):
        """Modo: Mapa estático 2D"""
        self.logger.info("\n" + "="*70)
        self.logger.info("MODO: MAPA ESTÁTICO 2D")
        self.logger.info("="*70)
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            
            self.logger.info("Cargando predicciones...")
            predictions = pd.read_csv(self.predictions_file)
            
            self.logger.info(f"Registros cargados: {len(predictions)}")
            
            # Crear mapa de probabilidad de helada
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Mapa 1: Temperatura mínima predicha
            scatter1 = ax1.scatter(
                predictions['lon'],
                predictions['lat'],
                c=predictions.get('tmin_pred', predictions.get('tmin', [0]*len(predictions))),
                cmap='RdYlBu_r',
                s=50,
                alpha=0.6
            )
            ax1.set_xlabel('Longitud')
            ax1.set_ylabel('Latitud')
            ax1.set_title('Temperatura Mínima Predicha (°C)')
            plt.colorbar(scatter1, ax=ax1)
            
            # Mapa 2: Probabilidad de helada
            if 'probabilidad_helada' in predictions.columns:
                scatter2 = ax2.scatter(
                    predictions['lon'],
                    predictions['lat'],
                    c=predictions['probabilidad_helada'],
                    cmap='Reds',
                    s=50,
                    alpha=0.6
                )
                ax2.set_xlabel('Longitud')
                ax2.set_ylabel('Latitud')
                ax2.set_title('Probabilidad de Helada')
                plt.colorbar(scatter2, ax=ax2)
            
            plt.suptitle('Predicción de Heladas - Región de Puno')
            plt.tight_layout()
            plt.show()
            
            self.logger.info("✓ Visualización estática completada")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error en visualización estática: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_animated(self):
        """Modo: Animación temporal"""
        self.logger.info("\n" + "="*70)
        self.logger.info("MODO: ANIMACIÓN TEMPORAL")
        self.logger.info("="*70)
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import numpy as np
            
            self.logger.info("Cargando predicciones...")
            predictions = pd.read_csv(self.predictions_file)
            
            # Agrupar por mes
            predictions['mes'] = pd.to_datetime(predictions['fecha']).dt.month
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            meses = sorted(predictions['mes'].unique())
            self.logger.info(f"Meses en datos: {meses}")
            
            def animate(frame):
                ax.clear()
                mes_actual = meses[frame % len(meses)]
                datos_mes = predictions[predictions['mes'] == mes_actual]
                
                scatter = ax.scatter(
                    datos_mes['lon'],
                    datos_mes['lat'],
                    c=datos_mes.get('probabilidad_helada', [0]*len(datos_mes)),
                    cmap='Reds',
                    s=50,
                    alpha=0.6,
                    vmin=0,
                    vmax=1
                )
                
                ax.set_xlabel('Longitud')
                ax.set_ylabel('Latitud')
                ax.set_title(f'Probabilidad de Helada - Mes {mes_actual}')
                ax.set_xlim(predictions['lon'].min(), predictions['lon'].max())
                ax.set_ylim(predictions['lat'].min(), predictions['lat'].max())
                plt.colorbar(scatter, ax=ax)
            
            anim = animation.FuncAnimation(fig, animate, frames=len(meses)*3, interval=500)
            plt.show()
            
            self.logger.info("✓ Animación completada")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error en animación: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_risk(self):
        """Modo: Mapa de riesgo con categorización"""
        self.logger.info("\n" + "="*70)
        self.logger.info("MODO: MAPA DE RIESGO CATEGORIZADO")
        self.logger.info("="*70)
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            
            self.logger.info("Cargando predicciones...")
            predictions = pd.read_csv(self.predictions_file)
            
            # Categorizar riesgo
            if 'probabilidad_helada' in predictions.columns:
                prob = predictions['probabilidad_helada']
                risk_categories = pd.cut(
                    prob,
                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
                )
            else:
                self.logger.warning("No hay columna de probabilidad, usando temperatura")
                risk_categories = pd.cut(
                    predictions.get('tmin', [0]*len(predictions)),
                    bins=5,
                    labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
                )
            
            # Mapear colores
            color_map = {
                'Muy Bajo': '#2ecc71',
                'Bajo': '#f1c40f',
                'Medio': '#f39c12',
                'Alto': '#e74c3c',
                'Muy Alto': '#c0392b'
            }
            
            colors = [color_map.get(cat, 'gray') for cat in risk_categories]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            scatter = ax.scatter(
                predictions['lon'],
                predictions['lat'],
                c=colors,
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
            
            ax.set_xlabel('Longitud')
            ax.set_ylabel('Latitud')
            ax.set_title('Mapa de Riesgo de Heladas - Región de Puno')
            
            # Leyenda personalizada
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color_map[cat], edgecolor='black', label=cat)
                for cat in ['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
            ]
            ax.legend(handles=legend_elements, loc='upper right', title='Nivel de Riesgo')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            self.logger.info("✓ Mapa de riesgo completado")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error en mapa de riesgo: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def run(self):
        """Ejecuta el visualizador en el modo especificado"""
        
        if not self.validate_requirements():
            self.logger.error("No se puede continuar sin archivos requeridos")
            return False
        
        mode_handlers = {
            "interactive": self.run_interactive,
            "static": self.run_static,
            "animated": self.run_animated,
            "risk": self.run_risk,
        }
        
        handler = mode_handlers.get(self.mode)
        if not handler:
            self.logger.error(f"Modo desconocido: {self.mode}")
            self.logger.info(f"Modos disponibles: {', '.join(mode_handlers.keys())}")
            return False
        
        return handler()


def main():
    """Función principal con argumentos CLI"""
    parser = argparse.ArgumentParser(
        description="Visualizador unificado para predicciones de heladas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python unified_visualizer.py                # Modo interactivo 3D (por defecto)
  python unified_visualizer.py --mode static  # Mapa estático 2D
  python unified_visualizer.py --mode animated # Animación temporal
  python unified_visualizer.py --mode risk    # Mapa de riesgo categorizado
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "static", "animated", "risk"],
        default="interactive",
        help="Modo de visualización (por defecto: interactive)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Modo silencioso"
    )
    
    args = parser.parse_args()
    
    visualizer = UnifiedVisualizer(mode=args.mode)
    success = visualizer.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
