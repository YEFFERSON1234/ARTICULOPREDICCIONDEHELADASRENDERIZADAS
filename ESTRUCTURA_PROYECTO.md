# Estructura del Proyecto de Predicción de Heladas

## Resumen de Organización Completa

Este documento describe la estructura final organizada del proyecto de predicción de heladas para Puno.

## Directorios Principales

```
ARTICULOPREDICCIONDEHELADASRENDERIZADAS/
├── src/                              # Scripts principales de Python
│   ├── evaluate.py                   # Evaluación de modelos
│   ├── evaluate_models.py            # Evaluación comparativa de modelos
│   ├── predict.py                    # Predicción de heladas
│   ├── train.py                      # Entrenamiento de modelos
│   ├── visualize.py                  # Visualización de resultados
│   ├── walk_forward_cv.py            # Validación cruzada walk-forward
│   └── generate_missing_plots.py     # Generación de gráficos faltantes
│
├── data_process/                     # Datos procesados para ML
│   ├── predictions.csv               # Predicciones generales
│   ├── predictions_pipeline.csv      # Predicciones del pipeline
│   ├── predictions_rf.csv            # Predicciones Random Forest
│   ├── predictions_mlp.csv           # Predicciones MLP
│   ├── predictions_ensemble.csv      # Predicciones Ensemble
│   ├── comparacion_modelos.csv       # Comparación entre modelos
│   ├── dataset_ML_final_completo.csv # Dataset final para ML
│   └── datos_heladas_puno_REAL.csv   # Datos reales de heladas
│
├── Predicciones/                     # Predicciones organizadas
│   ├── Futuras/                      # Predicciones futuras
│   │   ├── PREDICCIONES_FUTURO_30DIAS.csv
│   │   ├── prediccion_2026-06-03.csv
│   │   └── prediccion_manana.csv
│   ├── Historicas/                   # Predicciones históricas
│   │   └── predictions_historicas.csv
│   └── Por_Modelo/                   # Predicciones por cada modelo
│       ├── predictions_cnn1d.csv
│       ├── predictions_ensemble.csv
│       ├── predictions_holt_winters.csv
│       ├── predictions_lstm.csv
│       ├── predictions_maestro.csv
│       ├── predictions_mlp.csv
│       ├── predictions_prophet.csv
│       ├── predictions_rf.csv
│       ├── predictions_sarima_ann_hybrid.csv
│       ├── predictions_sarimax.csv
│       └── predictions_svm.csv
│
├── modelos/                          # Modelos y sus visualizaciones
│   ├── XGBoost/                      # Modelo XGBoost
│   │   └── graficos_resultados/      # Gráficos específicos XGBoost
│   ├── RandomForest/                # Modelo Random Forest
│   │   └── graficos_resultados/      # Gráficos específicos RF
│   ├── SVM/                          # Modelo SVM
│   │   └── graficos_resultados/      # Gráficos específicos SVM
│   ├── MLP/                          # Modelo MLP
│   │   └── graficos_resultados/      # Gráficos específicos MLP
│   ├── Ensemble/                     # Modelo Ensemble
│   │   └── graficos_resultados/      # Gráficos específicos Ensemble
│   ├── Comparacion_Resultados/       # Comparativas entre modelos
│   ├── Mapas_Renderizados/           # Mapas geográficos
│   └── README.md                     # Documentación de modelos
│
├── graficos_resultados/              # Gráficos generales
│   ├── 01_curva_roc.png
│   ├── 02_matriz_confusion.png
│   ├── [otros gráficos generales...]
│   └── graficosResultados2/          # Gráficos secundarios
│
├── data/                             # Datos brutos
│   └── data_merra2_puno/             # Datos MERRA-2 de Puno
│
├── Render/                           # Herramientas de renderizado
│   └── Mapa/                         # Renderizado de mapas
│
├── docs/                             # Documentación
│   └── media/                        # Imágenes de documentación
│
└── README.md                         # Documentación principal
```

## Cambios Realizados en la Organización

### 1. Predicciones Organizadas
- **Carpeta `Predicciones/`** organizada en tres subdirectorios:
  - `Futuras/`: Predicciones a futuro (30 días, fechas específicas)
  - `Historicas/`: Predicciones históricas validadas
  - `Por_Modelo/`: Predicciones desglosadas por cada modelo

### 2. Modelos Organizados
- **Carpeta `modelos/`** con subdirectorios por modelo:
  - `XGBoost/graficos_resultados/`
  - `RandomForest/graficos_resultados/`
  - `SVM/graficos_resultados/`
  - `MLP/graficos_resultados/` (nuevo)
  - `Ensemble/graficos_resultados/` (nuevo)
  - `Comparacion_Resultados/`
  - `Mapas_Renderizados/`
- **README.md** en `modelos/` para documentación

### 3. Scripts Actualizados
Se corrigieron las rutas en los siguientes scripts para eliminar dependencias de `limpiezadedatos/`:
- `src/visualize.py`: Rutas actualizadas para buscar en `data_process/` y `Predicciones/`
- `src/predict.py`: Rutas actualizadas para buscar en `data_process/`
- `src/evaluate.py`: Rutas actualizadas para buscar en `data_process/` y `Predicciones/`

### 4. Script de Generación de Gráficos
- **Nuevo script** `src/generate_missing_plots.py`:
  - Genera curvas ROC para modelos que no tienen
  - Genera matrices de confusión
  - Genera gráficos de dispersión
  - Enfocado en MLP y Ensemble

## Modelos Implementados

1. **XGBoost**: Gradient boosting con árboles de decisión
2. **Random Forest**: Bosque aleatorio de árboles de decisión
3. **SVM**: Support Vector Machine
4. **MLP**: Multi-Layer Perceptron (red neuronal)
5. **Ensemble**: Combinación de múltiples modelos
6. **Otros modelos avanzados**: CNN1D, LSTM, Prophet, SARIMA, Holt-Winters, etc.

## Archivos de Datos Principales

- `data_process/dataset_ML_final_completo.csv`: Dataset principal para ML
- `data_process/predictions.csv`: Predicciones consolidadas
- `Predicciones/Historicas/predictions_historicas.csv`: Predicciones históricas
- `Predicciones/Futuras/PREDICCIONES_FUTURO_30DIAS.csv`: Predicciones a 30 días

## Script de Utilidad

El script `src/generate_missing_plots.py` puede ejecutarse para generar gráficos faltantes:

```bash
python src/generate_missing_plots.py
```

Este script generará automáticamente:
- Curvas ROC
- Matrices de confusión
- Gráficos de dispersión

Para los modelos que tengan archivos de predicciones disponibles en `data_process/`.

## Estado de Organización

✅ **Completado:**
- Estructura de carpetas organizada
- Archivos de predicciones categorizados
- Scripts con rutas corregidas
- Directorios de modelos completos
- Documentación de estructura

✅ **Listo para uso:**
- Todos los scripts principales funcionan con las nuevas rutas
- Los datos están organizados lógicamente
- Las visualizaciones están estructuradas por modelo
- El proyecto está listo para análisis y documentación

## Notas

- Los archivos en `limpiezadedatos/` ya no son referenciados por los scripts principales
- Los datos principales están en `data_process/` y `Predicciones/`
- La estructura facilita el mantenimiento y escalabilidad del proyecto
- Cada modelo tiene su propio directorio para visualizaciones específicas