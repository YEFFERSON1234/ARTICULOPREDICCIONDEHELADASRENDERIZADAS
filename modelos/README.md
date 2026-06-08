# Estructura de Organización de Modelos

Este directorio contiene la organización de todos los modelos entrenados y sus respectivas visualizaciones.

## Estructura de Directorios

```
modelos/
├── XGBoost/
│   └── graficos_resultados/         # Gráficos específicos de XGBoost
├── RandomForest/
│   └── graficos_resultados/         # Gráficos específicos de Random Forest
├── SVM/
│   └── graficos_resultados/         # Gráficos específicos de SVM
├── MLP/
│   └── graficos_resultados/         # Gráficos específicos de MLP
├── Ensemble/
│   └── graficos_resultados/         # Gráficos específicos del Ensemble
├── Comparacion_Resultados/          # Gráficos comparativos entre modelos
└── Mapas_Renderizados/              # Mapas geográficos de predicciones
```

## Descripción de Directorios

- **XGBoost/**: Contiene los gráficos de rendimiento del modelo XGBoost
- **RandomForest/**: Contiene los gráficos de rendimiento del modelo Random Forest
- **SVM/**: Contiene los gráficos de rendimiento del modelo SVM
- **MLP/**: Contiene los gráficos de rendimiento del modelo MLP (Multi-Layer Perceptron)
- **Ensemble/**: Contiene los gráficos de rendimiento del modelo Ensemble (combinación de modelos)
- **Comparacion_Resultados/**: Gráficos que comparan el rendimiento entre diferentes modelos
- **Mapas_Renderizados/**: Mapas geográficos que muestran las predicciones espaciales

## Tipos de Gráficos

Cada directorio de modelo contiene típicamente:
- `curva_roc.png` - Curva ROC del modelo
- `matriz_confusion.png` - Matriz de confusión
- `grafico_dispersion.png` - Gráfico de dispersión de predicciones

El directorio de comparación contiene:
- Gráficos comparativos de métricas
- Perfiles de radar de rendimiento
- Curvas ROC superpuestas
- Tablas comparativas renderizadas
