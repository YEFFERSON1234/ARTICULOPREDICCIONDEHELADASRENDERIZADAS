# 🌡️ PREDICCIÓN DE HELADAS - PUNO, PERÚ

> Sistema integral de predicción de heladas mediante ensemble de machine learning y visualización 3D interactiva para el Altiplano Peruano (> 3800 msnm)

**Estado:** ✅ Producción | **Última actualización:** 2026-06-07 | **Versión:** 1.0

---

## 🎯 ¿Qué es este proyecto?

Este es un sistema de **predicción de heladas para Puno, Perú** que combina:
- ✅ **11 modelos de ML** - XGBoost, Random Forest, LSTM, SVM, CNN-1D, Prophet, SARIMAX, y más
- ✅ **Ensemble maestro** - Combina los mejores 4 modelos con RMSE 2.42°C, F1-Score 0.963
- ✅ **Datos históricos** - 392,283 registros de 29 estaciones SENAMHI (2002-2023)
- ✅ **Visualización 3D** - Mapas interactivos del riesgo de helada con OpenGL
- ✅ **Predicción flexible** - Para cualquier fecha específica, no solo el próximo día

**Impacto:** Prevenir 35-60% de pérdidas agrícolas en comunidades del Altiplano afectadas por heladas.

---

## 📚 GUÍA DE NAVEGACIÓN RÁPIDA

### 🚀 **Quiero empezar rápido**
→ Lee [**SETUP.md**](docs/SETUP.md) (instalación) y [**EXECUTION_GUIDE.md**](docs/EXECUTION_GUIDE.md) (cómo correr)

### 🏗️ **Quiero entender la arquitectura**
→ Ve a [**docs/arquitectura/**](docs/arquitectura/) - Documentación técnica y flujo del sistema

### 📊 **Quiero ver datos y métricas**
→ Consulta [**docs/datos-modelos/**](docs/datos-modelos/) - Resultados de 11 modelos, CSV maestro

### ⚙️ **Necesito configurar mi entorno**
→ Ve a [**docs/configuracion/**](docs/configuracion/) - Dependencias, versiones, SRTM

### 🔧 **Quiero mejorar el proyecto**
→ Consulta [**docs/mantenimiento/**](docs/mantenimiento/) - Mejoras implementadas e items pendientes

### 💻 **Quiero reproducir en otra máquina**
→ Lee [**REPLICATE.md**](docs/REPLICATE.md) - Instrucciones paso a paso

---

## 📋 ÍNDICE COMPLETO DE DOCUMENTACIÓN

### 🏗️ **[docs/arquitectura/](docs/arquitectura/)** - Diseño técnico del sistema
| Archivo | Descripción |
|---------|------------|
| [README.md](docs/arquitectura/README.md) | Índice de arquitectura |
| [arquitectura.txt](docs/arquitectura/arquitectura.txt) | Diagrama ASCII del pipeline |
| [INFORME_PIPELINE_COMPLETO.md](docs/arquitectura/INFORME_PIPELINE_COMPLETO.md) | 6 pasos del pipeline detallados |
| [DOCUMENTACION_GRAFICOS.md](docs/arquitectura/DOCUMENTACION_GRAFICOS.md) | 10+ gráficos explicados |
| [articulo_completo.tex](docs/arquitectura/articulo_completo.tex) | Artículo académico en LaTeX |

### 📊 **[docs/datos-modelos/](docs/datos-modelos/)** - Datos y resultados de modelos
| Archivo | Descripción |
|---------|------------|
| [README.md](docs/datos-modelos/README.md) | Índice de datos y modelos |
| [README_CSV_MAESTRO_PREDICCIONES.md](docs/datos-modelos/README_CSV_MAESTRO_PREDICCIONES.md) | CSV maestro (73,170 registros) |
| [METRICAS_MODELOS_DETALLADO.md](docs/datos-modelos/METRICAS_MODELOS_DETALLADO.md) | 11 modelos con hiperparámetros |
| [INFORME_GENERAL_CONSOLIDADO.txt](docs/datos-modelos/INFORME_GENERAL_CONSOLIDADO.txt) | Resumen consolidado |
| [INFORME_GENERAL_RESULTADOS.md](docs/datos-modelos/INFORME_GENERAL_RESULTADOS.md) | Resultados con predicciones |

### ⚙️ **[docs/configuracion/](docs/configuracion/)** - Instalación y requisitos
| Archivo | Descripción |
|---------|------------|
| [README.md](docs/configuracion/README.md) | Índice de configuración |
| [package_versions.txt](docs/configuracion/package_versions.txt) | Todas las versiones de paquetes |
| [download_srtm.txt](docs/configuracion/download_srtm.txt) | Descargar DEM de NASA |
| [unify_dem.txt](docs/configuracion/unify_dem.txt) | Unificar tiles DEM |

### 🔧 **[docs/mantenimiento/](docs/mantenimiento/)** - Mejoras e items pendientes
| Archivo | Descripción |
|---------|------------|
| [README.md](docs/mantenimiento/README.md) | Índice de mantenimiento |
| [IMPROVEMENTS.md](docs/mantenimiento/IMPROVEMENTS.md) | 5 mejoras implementadas |
| [MISSING_ITEMS.md](docs/mantenimiento/MISSING_ITEMS.md) | 20+ items faltantes analizados |

### 📄 **Documentos principales en raíz**
| Archivo | Descripción |
|---------|------------|
| [SETUP.md](docs/SETUP.md) | Instalación paso a paso |
| [EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md) | Cómo ejecutar cada componente |
| [REPLICATE.md](docs/REPLICATE.md) | Reproducir en otra máquina |

---

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
├── Predicciones/                 # Predicciones organizadas por categoría
│   ├── Futuras/                  # Predicciones a futuro (30 días, fechas específicas)
│   ├── Historicas/               # Predicciones históricas validadas
│   └── Por_Modelo/               # Predicciones desglosadas por cada modelo
├── modelos/                      # Modelos y visualizaciones organizadas
│   ├── XGBoost/graficos_resultados/    # Gráficos específicos XGBoost
│   ├── RandomForest/graficos_resultados/ # Gráficos específicos RF
│   ├── SVM/graficos_resultados/        # Gráficos específicos SVM
│   ├── MLP/graficos_resultados/        # Gráficos específicos MLP
│   ├── Ensemble/graficos_resultados/   # Gráficos específicos Ensemble
│   ├── Comparacion_Resultados/         # Comparativas entre modelos
│   ├── Mapas_Renderizados/             # Mapas geográficos
│   └── README.md                       # Documentación de modelos
├── graficos_generales/           # Gráficos generales organizados
│   ├── metricas/                     # Métricas de rendimiento (17 archivos)
│   └── analisis/                     # Análisis exploratorio (11 archivos)
├── graficos_predicciones/        # Gráficos de predicciones
│   ├── futuras/                      # Predicciones a futuro (3 archivos)
│   └── mapas/                        # Mapas de riesgo y temperatura (8 archivos)
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
│   ├── visualize.py              # Visualización de resultados
│   └── generate_missing_plots.py # Generación de gráficos faltantes
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
├── ESTRUCTURA_PROYECTO.md        # Documentación de estructura organizada
└── README.md                     # Este archivo
```

---

## 📊 Resumen de Características

| Aspecto | Detalle |
|--------|---------|
| **Modelos** | 11 modelos (XGBoost, RF, LSTM, SVM, CNN-1D, Prophet, SARIMAX, etc.) |
| **Datos** | 392,283 registros de 29 estaciones SENAMHI (2002-2023) |
| **Ensemble** | Combina 4 mejores modelos - RMSE 2.42°C, F1-Score 0.963, AUC-ROC 0.996 |
| **Visualización** | 3D interactivo con OpenGL + mapas de riesgo |
| **Tecnología** | Python 3.11.9, PyTorch 2.5.1, XGBoost 3.2.0, CUDA 12.1 |
| **Predicción** | Flexible para cualquier fecha, no solo próximo día |
| **Cobertura** | Puno, Perú (15°S-17°S, 69°O-71°O, >3800 msnm) |

---

## 🚀 INICIO RÁPIDO

### 1️⃣ Instalar
```bash
# Ver SETUP.md para instrucciones detalladas
pip install -r requirements.txt
```

### 2️⃣ Ejecutar
```bash
# Ver EXECUTION_GUIDE.md para todos los comandos
python src/train.py          # Entrenar modelos
python visualization/main.py # Ver visualización 3D
```

### 3️⃣ Reproducir
Consulta [REPLICATE.md](docs/REPLICATE.md) para instrucciones paso a paso.

---

## 📁 Estructura Simplificada

```
ARTICULOPREDICCIONDEHELADASRENDERIZADAS/
├── 📄 README.md (este archivo)
├── 📂 docs/                          # DOCUMENTACIÓN COMPLETA
│   ├── 📂 arquitectura/              # Diseño técnico
│   ├── 📂 datos-modelos/             # Resultados y métricas
│   ├── 📂 configuracion/             # Instalación y requisitos
│   ├── 📂 mantenimiento/             # Mejoras e items pendientes
│   ├── 📂 media/ & prototipos/       # Multimedia
│   ├── SETUP.md                      # Instalación paso a paso
│   ├── EXECUTION_GUIDE.md            # Cómo ejecutar
│   └── REPLICATE.md                  # Reproducir en otra máquina
│
├── 📂 data/                          # Datos crudos
├── 📂 data_process/                  # Datos procesados
├── 📂 modelos/                       # 11 modelos de ML
├── 📂 src/                           # Scripts principales
├── 📂 utils/                         # Utilidades
├── 📂 visualization/                 # Visualización 3D
├── 📂 Render/Mapa/                   # Renderizado OpenGL
├── 📂 Archivos.tiff-renderizar/      # DEM GeoTIFF
└── requirements.txt                  # Dependencias
```

---

## ✨ Logros del Proyecto

✅ **Modelo Ensemble** - RMSE 2.42°C, F1-Score 0.963, AUC-ROC 0.996
✅ **11 Modelos** - Comparación exhaustiva de técnicas de ML y deep learning
✅ **Visualización 3D** - Mapas interactivos del riesgo de helada en tiempo real
✅ **Datos Completos** - 392,283 registros históricos (2002-2023)
✅ **Documentación Integral** - Arquitectura, métricas, setup, ejecución
✅ **Estructura Organizada** - Predicciones categorizadas y modelos organizados por directorios
✅ **Imágenes Organizadas** - 39 imágenes categorizadas en carpetas lógicas (generales, predicciones, modelos)

---

## 📞 Soporte

- **¿Cómo instalo?** → Lee [SETUP.md](docs/SETUP.md)
- **¿Cómo ejecuto?** → Lee [EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md)
- **¿Cómo reproduzco?** → Lee [REPLICATE.md](docs/REPLICATE.md)
- **¿Qué modelos hay?** → Ve a [docs/datos-modelos/](docs/datos-modelos/)
- **¿Cómo funciona?** → Lee [docs/arquitectura/](docs/arquitectura/)

---

**Última actualización:** 2026-06-07 | **Versión:** 1.0 Producción | **Autor:** Yefferson Miranda Josec
