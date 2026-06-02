# Guía de Ejecución del Proyecto

Esta guía te muestra paso a paso cómo ejecutar el proyecto de predicción de heladas en Puno y ver los resultados.

## 🚀 Requisitos Previos

### 1. Instalar Python y dependencias
```powershell
# Crear entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r ../requirements.txt
```

### 2. Verificar instalación
```powershell
python --version  # Debe ser Python 3.11.9
pip list         # Verificar paquetes instalados
```

## 📊 Flujo de Ejecución Completo

### Paso 1: Procesar datos ERA5 (Opcional)
Si tienes archivos `.nc` de ERA5 en `data/datos_era5_puno/`:

```powershell
python utils/process_csv.py
```

**Resultado**: Genera `data/era5_procesado_maestro.csv`

### Paso 2: Unificar datos SENAMHI + ERA5
Abre el notebook `unificar_datos.ipynb` en Jupyter y ejecuta todas las celdas:

```powershell
jupyter notebook unificar_datos.ipynb
```

**Resultado**: Genera `data_process/dataset_ML_final_completo.csv`

### Paso 3: Entrenar modelos
```powershell
# Opción 1: Usar el script en src/
python src/train.py

# Opción 2: Usar el modelo directamente
python modelos/xgboost_model.py
```

**Resultado**: Genera `data_process/predictions.csv`

### Paso 4: Generar predicciones finales
```powershell
python data_process/generate_final_predictions.py
```

**Resultado**: Actualiza `data_process/predictions.csv` con formato correcto

### Paso 5: Preparar el terreno 3D (DEM)
```powershell
python Render/Mapa/convert_tiles_to_csv.py
```

**Resultado**: Genera `Render/Mapa/dem_puno_render.csv.gz`

### Paso 6: Visualizar resultados 3D

#### Opción A: Visualizador principal (Recomendado)
```powershell
python visualization/main.py
```

**Controles**:
- Mouse arrastrar: rotar
- Scroll / +/-: zoom
- Flechas: rotar cámara
- Espacio: auto-rotación
- R: reset cámara
- ESC: salir

#### Opción B: Visualizador estático
```powershell
python Render/Mapa/draw_from_csv.py
```

#### Opción C: Visualizador con riesgo de helada
```powershell
python Render/Mapa/visualizer_with_risk.py
```

#### Opción D: Visualizador animado
```powershell
python Render/Mapa/animated_visualizer.py
```

### Paso 7: Ver prototipo web (Opcional)
Abre en navegador:
```
docs/prototipos/web_visualizer.html
```

## 📈 Ver Resultados

### Archivos de resultados generados:

1. **Predicciones**: `data_process/predictions.csv`
   - Contiene: lat, lon, fecha, prob_helada
   - Formato: CSV

2. **Terreno 3D**: `Render/Mapa/dem_puno_render.csv.gz`
   - Contiene: longitud, latitud, elevación
   - Formato: CSV comprimido

3. **Vértices del terreno**: `Render/Mapa/terrain_vertices.npy`
   - Contiene: vértices normalizados para OpenGL
   - Formato: NumPy array

### Para ver predicciones en Python:
```python
import pandas as pd

# Cargar predicciones
predictions = pd.read_csv('data_process/predictions.csv')

# Ver primeras filas
print(predictions.head())

# Ver estadísticas
print(predictions['prob_helada'].describe())

# Ver predicciones de alto riesgo
high_risk = predictions[predictions['prob_helada'] > 0.7]
print(f"Predicciones de alto riesgo: {len(high_risk)}")
```

## 🔧 Solución de Problemas

### Error: "No module named 'sklearn'"
```powershell
pip install scikit-learn
```

### Error: "OpenGL/Pygame no disponible"
```powershell
pip install PyOpenGL pygame
```

### Error: "No se encontró data_process/predictions.csv"
Asegúrate de haber ejecutado primero:
```powershell
python modelos/xgboost_model.py
python data_process/generate_final_predictions.py
```

### Error: "No se encontró dem_puno_render.csv.gz"
Ejecuta primero:
```powershell
python Render/Mapa/convert_tiles_to_csv.py
```

## 📝 Notas Importantes

1. **Orden de ejecución**: Sigue los pasos en orden numerado
2. **Datos necesarios**: Asegúrate de tener los datos SENAMHI en `data_process/datos_heladas_puno_REAL.csv`
3. **Tiles DEM**: Los archivos GeoTIFF deben estar en `Archivos.tiff-renderizar/`
4. **Memoria**: Algunos procesos requieren mucha RAM (especialmente con datasets grandes)
5. **Tiempo**: El entrenamiento de modelos puede tardar varios minutos

## 🎯 Resultados Esperados

Al completar todos los pasos deberías poder:
- ✅ Ver predicciones de heladas en `data_process/predictions.csv`
- ✅ Visualizar el terreno 3D de Puno con OpenGL
- ✅ Ver mapa de calor de riesgo de heladas sobre el terreno
- ✅ Interactuar con la visualización 3D (rotar, zoom, etc.)

## 📞 Ayuda Adicional

- Para más detalles técnicos, consulta `docs/README.md`
- Para ver versiones de paquetes, consulta `docs/package_versions.txt`
- Para instrucciones de descarga de datos, consulta `docs/download_srtm.txt`
