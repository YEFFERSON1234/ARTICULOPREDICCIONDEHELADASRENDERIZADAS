# Documentación del Proyecto

Esta carpeta contiene documentación técnica y recursos de apoyo para el proyecto de predicción de heladas en Puno.

## 📁 Estructura

### Documentación Técnica
- **`unify_dem.txt`**: Diagrama ASCII que explica el proceso de unificación de tiles DEM (Digital Elevation Model) en un solo archivo unificado para la región de Puno.

- **`package_versions.txt`**: Lista detallada de versiones de todos los paquetes Python utilizados (PyTorch 2.5.1, CUDA 12.1, XGBoost 3.2.0, etc.) e incluye código LaTeX para el artículo académico.

- **`download_srtm.txt`**: Instrucciones paso a paso para descargar los tiles SRTM de NASA Earthdata necesarios para el proyecto.

### Prototipos
- **`prototipos/web_visualizer.html`**: Prototipo funcional de visualización 3D interactiva web del terreno de Puno con riesgo de heladas. Implementa terreno 3D con controles interactivos (día del año, rotación, animación).

### Multimedia
- **`media/map_animation.gif`**: Animación/grabación de visualización del mapa de heladas.
- **`media/map_animation_animated.gif`**: Animación/grabación adicional del mapa con animación.

## 📝 Uso

### Para reproducir el entorno:
```bash
pip install -r ../requirements.txt
```

Consulte `package_versions.txt` para ver las versiones específicas de cada paquete.

### Para descargar datos DEM:
Siga las instrucciones en `download_srtm.txt` para obtener los tiles SRTM de NASA Earthdata.

### Para ver el prototipo web:
Abra `prototipos/web_visualizer.html` en un navegador web para ver la visualización 3D interactiva.
