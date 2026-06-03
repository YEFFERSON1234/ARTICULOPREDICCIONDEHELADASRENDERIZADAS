"""
Configuración centralizada del proyecto
"""
import os
from pathlib import Path

# Directorio base del proyecto
PROJECT_ROOT = Path(__file__).parent.absolute()

# Directorios de datos
DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESS_DIR = PROJECT_ROOT / "data_process"
ARCHIVOS_TIFF_DIR = PROJECT_ROOT / "Archivos.tiff-renderizar"

# Archivos de datos importantes
SENAMHI_CSV = DATA_PROCESS_DIR / "datos_heladas_puno_REAL.csv"
ERA5_MAESTRO_CSV = DATA_DIR / "era5_procesado_maestro.csv"
DATASET_FINAL_ML = DATA_PROCESS_DIR / "dataset_ML_final_completo.csv"
PREDICTIONS_CSV = DATA_PROCESS_DIR / "predictions_pipeline.csv"

# Directorios de modelos y resultados
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "resultados"
LOGS_DIR = PROJECT_ROOT / "logs"

# Directorios de visualización
VISUALIZATION_DIR = PROJECT_ROOT / "visualization"
RENDER_DIR = PROJECT_ROOT / "Render" / "Mapa"

# Parámetros de entrenamiento
TRAIN_TEST_YEAR_SPLIT = True  # Si True, usa el último año como test
TRAIN_PARAMS = {
    "n_estimators": 300,
    "max_depth": 7,
    "learning_rate": 0.04,
    "tree_method": "hist"
}

# Features para el modelo
FEATURE_COLS = [
    'lat', 'lon', 'day_of_year', 'month', 'precip', 'tmax',
    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3'
]

# Coordenadas aproximadas de Puno, Perú
PUNO_LAT = -15.5
PUNO_LON = -70.1
PUNO_BOUNDS = {
    'lat_min': -17.0,
    'lat_max': -14.0,
    'lon_min': -71.5,
    'lon_max': -68.5
}

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Crear directorios necesarios
for dir_path in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
