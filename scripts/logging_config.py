"""
Configuración de logging estructurado para el proyecto
"""
import logging
import logging.handlers
from pathlib import Path
import scripts.config as config

def setup_logging(name=__name__, log_level=config.LOG_LEVEL):
    """
    Configura logging estructurado con archivos y consola
    
    Args:
        name: Nombre del logger (típicamente __name__)
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # Evitar agregar handlers duplicados
    if logger.hasHandlers():
        return logger
    
    # Formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo con rotación
    log_file = config.LOGS_DIR / f"{name.split('.')[-1]}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Logger global del proyecto
project_logger = setup_logging("frost_prediction")
