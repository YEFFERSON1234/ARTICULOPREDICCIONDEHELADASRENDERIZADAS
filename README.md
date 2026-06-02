# Predicción de Heladas Renderizadas en Puno

Este proyecto de machine learning predice heladas en la región de Puno, Perú, combinando datos meteorológicos históricos (SENAMHI), datos satelitales (ERA5, MODIS) y visualización 3D interactiva con OpenGL.

## 📁 Estructura del proyecto

```
ARTICULOPREDICCIONDEHELADASRENDERIZADAS/
├── data/                          # Datos crudos
│   ├── datos_era5_puno/          # Archivos .nc de ERA5
│   ├── datos_senami_puno/        # Datos históricos SENAMHI (.txt)
│   └── era5_procesado_maestro.csv # ERA5 procesado
├── data_process/                 # Datos procesados
│   ├── datos_heladas_puno_REAL.csv  # Datos históricos SENAMHI
│   ├── dataset_ML_final_completo.csv  # Dataset unificado (SENAMHI + ERA5)
│   ├── predictions.csv           # Predicciones finales
│   └── lstm_puno_v1.pth          # Modelo LSTM entrenado
├── modelos/                      # Modelos de ML
│   ├── xgboost_model.py          # Modelo XGBoost principal
│   ├── random_forest.py           # Random Forest
│   ├── lstm_pytorch.py           # LSTM en PyTorch
│   ├── cnn_1d_model.py           # CNN 1D
│   ├── SVM.py                    # Support Vector Machine
│   ├── ensamble.py               # Ensemble de modelos
│   ├── holt_winters_model.py     # Holt-Winters
│   ├── mlp_model.py              # MLP
│   ├── prophet_model.py          # Prophet
│   ├── sarima_ann_hybrid.py      # SARIMA-ANN Hybrid
│   └── sarimax_model.py          # SARIMAX
├── src/                          # Scripts de entrenamiento/evaluación
│   ├── train.py                  # Entrenamiento de modelos
│   ├── predict.py                # Predicción
│   ├── evaluate.py               # Evaluación de métricas
│   └── visualize.py              # Visualización de resultados
├── utils/                        # Scripts de utilidad
│   ├── process_csv.py            # Procesa archivos ERA5 .nc a CSV
│   └── download_modis.py         # Descarga datos MODIS (Google Earth Engine)
├── visualization/                # Visualización 3D
│   ├── main.py                   # Integración final ML + OpenGL
│   ├── renderer.py               # Motor de renderizado 3D
│   └── terrain_mesh.py           # Generación de malla del terreno
├── Render/Mapa/                  # Archivos de renderizado
│   ├── convert_tiles_to_csv.py   # Convierte DEM a CSV
│   ├── draw_from_csv.py          # Visualizador 3D estático
│   ├── visualizer_with_risk.py   # Visualizador con riesgo de helada
│   ├── animated_visualizer.py    # Visualizador animado
│   ├── dem_puno_render.csv.gz    # DEM comprimido (5.9 MB)
│   ├── terrain_vertices.npy      # Vértices del terreno
│   └── help.md                   # Documentación de visualización
├── Archivos.tiff-renderizar/     # Archivos DEM GeoTIFF
│   ├── s15_w069_1arc_v3.tif
│   ├── s15_w070_1arc_v3.tif
│   ├── s15_w071_1arc_v3.tif
│   ├── s16_w069_1arc_v3.tif
│   ├── s16_w070_1arc_v3.tif
│   ├── s16_w071_1arc_v3.tif
│   ├── s17_w069_1arc_v3.tif
│   ├── s17_w070_1arc_v3.tif
│   └── s17_w071_1arc_v3.tif
├── docs/                         # Documentación técnica
│   ├── unify_dem.txt             # Diagrama de unificación DEM
│   ├── package_versions.txt      # Versiones de paquetes y LaTeX
│   ├── download_srtm.txt         # Instrucciones descarga SRTM
│   ├── prototipos/               # Prototipos y demos
│   │   └── web_visualizer.html   # Visualizador 3D web
│   └── media/                    # Archivos multimedia
│       ├── map_animation.gif
│       ├── map_animation_animated.gif
│       └── website_credential.png
├── unificar_datos.ipynb          # Notebook para unificar datos SENAMHI + ERA5
├── arquitectura.txt              # Diagrama de arquitectura del sistema
└── README.md                     # Este archivo
```

## 🎯 Objetivos del proyecto

- **Predicción de temperatura mínima** (`tmin_pred`) usando modelos de machine learning
- **Clasificación de riesgo de helada** (`helada`: 1 si tmin ≤ 0°C, 0 si no)
- **Visualización 3D interactiva** del terreno de Puno con mapas de riesgo
- **Integración de múltiples fuentes de datos**: SENAMHI (estaciones), ERA5 (reanálisis), MODIS (satélite)

## 🧠 Variables del modelo

### Variables de entrada (features)
- **Geográficas**: `lat`, `lon` (coordenadas de la estación)
- **Temporales**: `day_of_year`, `month`, `year`
- **Meteorológicas (SENAMHI)**: `precip`, `tmax`, `tmin`
- **Lags temporales**: `tmin_lag_1`, `tmin_lag_2`, `tmin_lag_3` (tmin de los 3 días anteriores)
- **ERA5 (opcional)**: `temp_2m_era5`, `precip_era5`, `presion_era5`, `dew_point_era5`

### Variables de salida (predictions)
- `tmin_pred`: temperatura mínima predicha (°C)
- `probabilidad_helada`: probabilidad de ocurrencia de helada (0-1)
- `helada`: clasificación binaria (1 = helada, 0 = no helada)

## 🛠️ Instalación de dependencias

### Requisitos
- Python 3.11.9
- Windows (PowerShell)

### Pasos de instalación

```powershell
# 1. Crear entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Actualizar pip
pip install --upgrade pip

# 3. Instalar PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar Machine Learning y Ciencia de Datos
pip install numpy==2.4.3 pandas==3.0.2 scipy==1.17.1 xgboost==3.2.0 scikit-learn==1.8.0

# 5. Instalar Geoespacial (para archivos GeoTIFF)
pip install rasterio==1.4.4 affine==2.4.0 click==8.3.3 cligj==0.7.2 click-plugins==1.1.1.2

# 6. Instalar Visualización
pip install matplotlib==3.10.9 seaborn==0.13.2
pip install opencv-python==4.13.0.92 pillow==12.1.1
pip install pyopengl glfw

# 7. Dependencias adicionales
pip install joblib==1.5.3 threadpoolctl==3.6.0 networkx==3.6.1 sympy==1.13.1 jinja2==3.1.6 fsspec==2026.2.0
pip install xarray geemap earthengine-api
```

## 🚀 Flujo de trabajo

### 1. Procesar datos ERA5 (opcional)
Si tienes archivos `.nc` de ERA5 en `data/datos_era5_puno/`:

```powershell
python utils/process_csv.py
```

Esto genera `data/era5_procesado_maestro.csv`

### 2. Unificar datos SENAMHI + ERA5
Abre el notebook `unificar_datos.ipynb` y ejecuta las celdas para:
- Cargar datos de SENAMHI (`data_process/datos_heladas_puno_REAL.csv`)
- Cargar datos de ERA5 procesados
- Unificar por fecha y coordenadas
- Imputar valores faltantes
- Guardar `data_process/dataset_ML_final_completo.csv`

### 3. Entrenar modelos
```powershell
# Opción 1: Usar el script en src/
python src/train.py

# Opción 2: Usar el modelo directamente
python modelos/xgboost_model.py
```

**Nota**: El script `modelos/xgboost_model.py` actualmente busca datos en `limpiezadedatos/` pero deberías actualizarlo para usar `data_process/`.

### 4. Descargar datos MODIS (opcional)
Requiere configurar un ID de proyecto de Google Cloud:

1. Edita `utils/download_modis.py` y cambia `EE_PROJECT_ID = "PON_AQUI_TU_ID_DE_PROYECTO"` por tu ID real
2. Autentícate con Google Earth Engine
3. Ejecuta:

```powershell
python utils/download_modis.py
```

Esto genera `data/modis/csv/modis_processed_direct.csv`

### 5. Visualizar resultados 3D

#### Preparar el terreno (DEM)
```powershell
python Render/Mapa/convert_tiles_to_csv.py
```
Genera `Render/Mapa/dem_puno_render.csv.gz`

#### Visualizador principal (integración ML + OpenGL)
```powershell
python visualization/main.py
```

#### Visualizador estático
```powershell
python Render/Mapa/draw_from_csv.py
```

#### Visualizador con riesgo de helada
```powershell
python Render/Mapa/visualizer_with_risk.py
```

#### Visualizador animado
```powershell
python Render/Mapa/animated_visualizer.py
```

**Controles del visualizador 3D**:
- Mouse arrastrar: rotar
- +/-: zoom
- Flechas: mover cámara

## 📊 Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENTRENAMIENTO (Python)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  datos_heladas_puno_REAL.csv  →  XGBoost/LSTM/RF  →  Ensemble               │
│                                    ↓                                         │
│                          predictions.csv                                     │
│                    (station, lat, lon, temp, riesgo_helada)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROCESAMIENTO DEL TERRENO                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Tiles DEM (s15_w069.tif, s16_w070.tif, ...)  →  convertir_tiles_a_csv.py   │
│                                    ↓                                         │
│                    dem_puno_render.csv.gz (5.9 MB)                            │
│                    (longitud, latitud, elevacion)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VISUALIZACIÓN 3D (OpenGL)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  dem_puno_render.csv.gz  →  Malla 3D del terreno                             │
│                      +                                                       │
│  predictions.csv  →  Color según riesgo de helada                           │
│                      ↓                                                       │
│              Renderizado en tiempo real (interactivo)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📈 Modelos disponibles

| Modelo | Archivo | Descripción |
|--------|---------|-------------|
| XGBoost | `modelos/xgboost_model.py` | Modelo principal (regresión + clasificación) |
| Random Forest | `modelos/random_forest.py` | Bosque aleatorio |
| LSTM | `modelos/lstm_pytorch.py` | Red neuronal recurrente en PyTorch |
| CNN 1D | `modelos/cnn_1d_model.py` | Red neuronal convolucional 1D |
| SVM | `modelos/SVM.py` | Support Vector Machine |
| Ensemble | `modelos/ensamble.py` | Combinación de múltiples modelos |
| Holt-Winters | `modelos/holt_winters_model.py` | Suavizamiento exponencial |
| MLP | `modelos/mlp_model.py` | Perceptrón multicapa |
| Prophet | `modelos/prophet_model.py` | Modelo Prophet de Facebook |
| SARIMA-ANN | `modelos/sarima_ann_hybrid.py` | Híbrido SARIMA-ANN |
| SARIMAX | `modelos/sarimax_model.py` | SARIMA con variables exógenas |

## 🔧 Correcciones aplicadas

1. ✅ **Reorganización de scripts**: Scripts movidos a carpetas lógicas (`utils/` y `visualization/`)
2. ✅ **Reorganización de documentación**: Archivos de `HELPimagenes/` movidos a `docs/` con estructura organizada
3. ✅ **Renombramiento de archivos**: Todos los archivos renombrados a snake_case consistente
4. ✅ **Eliminado archivo duplicado**: `requeriment.txt` eliminado (mantenido `requirements.txt`)
5. ✅ **Corregido .gitignore**: Error en línea 11 corregido (`*.pyc` y `*.tif` separados)
6. ✅ **Eliminadas credenciales expuestas**: Archivo `HELPimagenes/credencial.md` eliminado
7. ✅ **Corregido `src/train.py`**: Eliminada referencia a directorio inexistente `limpiezadedatos/`
8. ✅ **Actualizadas rutas en scripts**: Todos los scripts movidos tienen rutas relativas correctas
9. ✅ **Creados archivos `__init__.py`**: Para que `utils/` y `visualization/` funcionen como paquetes
10. ✅ **Documentación organizada**: Creada carpeta `docs/` con subcarpetas `prototipos/` y `media/`
11. ✅ **Referencias actualizadas**: README.md y scripts actualizados con nuevos nombres de archivos

## 📝 Archivos de datos

### Datos históricos SENAMHI
- `data_process/datos_heladas_puno_REAL.csv` (27 MB): Datos de estaciones meteorológicas de Puno (2002-2023)
- Columnas: `year`, `month`, `day`, `precip`, `tmax`, `tmin`, `estacion`, `lat`, `lon`, `zona`, `departamento`, `fecha`, `amp_termica`, `helada`

### Dataset unificado
- `data_process/dataset_ML_final_completo.csv` (61 MB): Combina SENAMHI + ERA5
- Columnas adicionales: `latitude`, `longitude`, `temp_2m_era5`, `precip_era5`, `presion_era5`, `dew_point_era5`

### Datos ERA5
- `data/datos_era5_puno/`: Archivos `.nc` mensuales (2015-2018)
- Variables: `t2m` (temperatura 2m), `tp` (precipitación), `sp` (presión), `d2m` (punto de rocío)

## 🌐 Cobertura geográfica

Los archivos DEM cubren la región de Puno:
- **Latitud**: 15°S a 17°S
- **Longitud**: 69°O a 71°O
- **Tiles disponibles**: s15_w069, s15_w070, s15_w071, s16_w069, s16_w070, s16_w071, s17_w069, s17_w070, s17_w071

## 📚 Referencias útiles

- `docs/EXECUTION_GUIDE.md`: **Guía paso a paso para ejecutar el proyecto y ver resultados**
- `docs/MISSING_ITEMS.md`: **Análisis profundo de faltantes del proyecto y priorización de mejoras**
- `arquitectura.txt`: Diagrama detallado del flujo de entrenamiento y renderizado
- `docs/README.md`: Documentación técnica completa del proyecto
- `docs/unify_dem.txt`: Instrucciones para unificar tiles DEM
- `docs/package_versions.txt`: Versiones de paquetes y código LaTeX para artículo
- `docs/download_srtm.txt`: Guía para descargar datos SRTM de NASA
- `docs/prototipos/web_visualizer.html`: Prototipo de visualización 3D web
- `Render/Mapa/help.md`: Documentación de visualización OpenGL
- `Archivos.tiff-renderizar/info.md`: Información sobre cobertura de tiles DEM

## ✅ Estado actual

- ✅ Datos SENAMHI procesados y listos
- ✅ Scripts de entrenamiento funcionales
- ✅ Múltiples modelos implementados (XGBoost, RF, LSTM, SVM)
- ✅ Visualización 3D con OpenGL
- ✅ Integración con datos ERA5
- ✅ Estructura de proyecto organizada y documentada
- ⚠️ Requiere configuración para descarga MODIS

## 🤝 Contribución

Para mejorar el proyecto:
- Actualizar rutas de archivos en scripts
- Agregar más variables de ERA5 al dataset
- Implementar validación cruzada temporal
- Añadir métricas de evaluación (RMSE, MAE, matriz de confusión)
- Generar gráficos de tmin real vs predicho
