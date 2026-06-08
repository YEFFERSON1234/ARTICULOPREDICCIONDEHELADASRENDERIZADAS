# Gráficos Generales del Proyecto

Este directorio contiene los gráficos generales del proyecto de predicción de heladas, organizados por categoría.

## Estructura de Directorios

```
graficos_generales/
├── metricas/                       # Métricas de rendimiento de modelos
│   ├── 01_curva_roc.png
│   ├── 02_matriz_confusion.png
│   ├── 03_scatter_plot.png
│   ├── 04_precision_recall_curve.png
│   ├── 05_metricas_por_fold.png
│   ├── 06_comparacion_modelos.png
│   ├── 07_roc_comparacion_modelos.png
│   ├── 08_resumen_metricas.png
│   ├── 17_metricas_rendimiento.png
│   ├── 20_matrices_confusion_agregadas.png
│   ├── comparacion_metricas.png
│   ├── curva_roc_v2.png
│   ├── matriz_confusion_v2.png
│   ├── metricas_comparativas_completo.png
│   ├── metricas_mejor_modelo.png
│   ├── metricas_radar_chart.png
│   ├── metricas_tabla_comparativa.png
│   └── xgboost_results.png
│
└── analisis/                      # Análisis de datos y patrones
    ├── 09_analisis_temporal.png
    ├── 14_heatmap_riesgo_heladas.png
    ├── 15_distribucion_riesgos.png
    ├── 16_distribucion_probabilidades.png
    ├── 18_analisis_estaciones.png
    ├── 19_analisis_temporal.png
    ├── 21_variabilidad_desviacion.png
    ├── distribucion_errores.png
    ├── importancia_variables.png
    ├── pesos_ensemble.png
    ├── probabilidad_helada_por_modelo.png
    └── temperatura_real_vs_predicha.png
```

## Descripción de Contenidos

### metricas/
Contiene gráficos relacionados con el rendimiento y evaluación de modelos:
- **Curvas ROC**: Análisis de sensibilidad y especificidad
- **Matrices de confusión**: Evaluación de clasificación binaria
- **Comparaciones de modelos**: Tablas y gráficos comparativos
- **Métricas por fold**: Resultados de validación cruzada
- **Scatter plots**: Análisis de dispersión de predicciones

### analisis/
Contiene gráficos de análisis exploratorio y patrones:
- **Análisis temporal**: Evolución de heladas en el tiempo
- **Heatmaps**: Mapas de calor de riesgo de heladas
- **Distribuciones**: Análisis de probabilidades y riesgos
- **Importancia de variables**: Features más relevantes
- **Pesos de ensemble**: Contribución de cada modelo al ensemble
- **Análisis de estaciones**: Comparación entre diferentes estaciones

## Notas

- Los gráficos están organizados para facilitar la navegación y análisis
- Los nombres de archivos incluyen prefijos numéricos para mantener orden
- Los archivos con sufijo "_v2" indican versiones alternativas o actualizadas