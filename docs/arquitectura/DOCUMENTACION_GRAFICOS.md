# Documentación de Gráficos - Predicción de Heladas

## Resumen de Visualizaciones Generadas

Se han generado **10 gráficos interactivos y análiticos** que presentan los resultados de los modelos de predicción de heladas en Puno.

---

## 📊 Gráficos Generados

### 1. **01_curva_roc.png** - Curva ROC (Receiver Operating Characteristic)
**Descripción:** 
- Muestra la relación entre la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR)
- Incluye múltiples modelos para comparación
- El área bajo la curva (AUC) indica la capacidad discriminativa del modelo
- **Interpretación:** Cuanto más cerca esté la curva de la esquina superior izquierda, mejor es el modelo
- **Resultado:** AUC = 0.9620 (Excelente rendimiento)

**Modelos incluidos:**
- Modelo Principal: 0.9620
- LSTM: Incluido
- MLP: Incluido
- Prophet: Incluido
- SARIMAX: Incluido
- CNN1D: Incluido
- Ensemble: Incluido
- Random Forest: Incluido

---

### 2. **02_matriz_confusion.png** - Matriz de Confusión
**Descripción:**
- Matriz de 2x2 que muestra la distribución de predicciones correctas e incorrectas
- Incluye dos versiones: valores absolutos y porcentajes

**Resultados:**
- **Verdaderos Positivos (TP):** 484 - Heladas correctamente identificadas
- **Verdaderos Negativos (TN):** 325 - No heladas correctamente identificadas
- **Falsos Positivos (FP):** 50 - Falsas alarmas (predijo helada, pero no hubo)
- **Falsos Negativos (FN):** 47 - Heladas no detectadas

**Métricas derivadas:**
- Precisión: 90.64% (de las heladas predichas, 90.64% eran correctas)
- Recall/Sensibilidad: 91.15% (detectó el 91.15% de todas las heladas)

---

### 3. **03_scatter_plot.png** - Gráfico de Dispersión
**Descripción:**
Dos visualizaciones:
- **Izquierda:** Scatter plot mostrando probabilidades predichas vs valores reales (0 = Sin helada, 1 = Con helada)
- **Derecha:** Histograma de distribución de probabilidades separado por clase

**Interpretación:**
- Los puntos rojo en alto (> 0.5) = Heladas correctamente predichas
- Los puntos azules en bajo (< 0.5) = No heladas correctamente predichas
- La línea punteada negra en 0.5 es el umbral de decisión

---

### 4. **04_precision_recall_curve.png** - Curva Precision-Recall
**Descripción:**
- Alternativa a ROC más útil cuando hay desbalance de clases
- Muestra el trade-off entre Precisión y Recall
- **Precisión:** Porcentaje de predicciones positivas correctas
- **Recall:** Porcentaje de positivos reales identificados

**Interpretación:**
- Área bajo la curva indica el balance entre precisión y cobertura
- Útil para entender el rendimiento en situaciones donde los falsos positivos/negativos tienen diferentes costos

---

### 5. **05_metricas_por_fold.png** - Métricas por Fold (Validación Cruzada)
**Descripción:**
Cuatro gráficos que muestran la estabilidad del modelo:

- **F1 Score por Fold:** Promedio = 0.9109 ± 0.0047 (Muy estable)
- **AUC-ROC por Fold:** Promedio = 0.9742 ± 0.0037 (Excelente)
- **Brier Score por Fold:** Promedio = 0.0684 ± 0.0023 (Bajo error)
- **Distribución de Muestras:** Muestra el crecimiento en tamaño de entrenamiento

**Interpretación:**
- La baja desviación estándar indica que el modelo es estable y generaliza bien
- Los scores son consistentes entre folds

---

### 6. **06_comparacion_modelos.png** - Comparación de Modelos
**Descripción:**
- Compara métricas entre XGBoost y Random Forest
- Muestra 6 métricas principales:
  - F1-Score
  - AUC-ROC
  - AUC-PR
  - Brier Score
  - CSI (Critical Success Index)
  - Accuracy

**Resultados:**
- XGBoost: Mejor en la mayoría de métricas
- Random Forest: Competitivo y más interpretable

---

### 7. **07_roc_comparacion_modelos.png** - Comparación ROC de Todos los Modelos
**Descripción:**
- Superpone todas las curvas ROC de los diferentes modelos
- Permite visualizar fácilmente cuál es el mejor modelo

**Ranking de AUC:**
1. Modelo Principal: 0.9620
2. Ensemble: Incluido en comparación
3. CNN1D: Incluido
4. Prophet: Incluido
5. LSTM: Incluido
6. MLP: Incluido
7. SARIMAX: Incluido
8. Random Forest: Incluido

---

### 8. **08_resumen_metricas.png** - Resumen de Métricas Principales
**Descripción:**
Gráfico de barras con las 5 métricas más importantes:

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 0.8929 (89.29%) |
| **Precision** | 0.9064 (90.64%) |
| **Recall** | 0.9115 (91.15%) |
| **F1-Score** | 0.9089 (90.89%) |
| **AUC-ROC** | 0.9620 (96.20%) |

**Interpretación:**
Todos los valores están en rango excelente (> 0.89), indicando un modelo de alta calidad

---

### 9. **09_analisis_temporal.png** - Análisis Temporal
**Descripción:**
Dos visualizaciones temporales:

- **Izquierda:** Precisión mensual (línea con área rellena)
- **Derecha:** Comparación heladas reales vs predichas por mes

**Interpretación:**
- Muestra cómo varía el rendimiento del modelo a lo largo del tiempo
- Identifica periodos con mejor/peor precisión
- Compara el número de heladas reales vs predichas

---

### 10. **10_violin_plot.png** - Distribución de Probabilidades (Violin Plot)
**Descripción:**
- Muestra la distribución de probabilidades predichas para cada clase
- La distribución de violín es más informativa que un simple box plot

**Interpretación:**
- **Clase Sin Helada:** Concentrada cerca de 0
- **Clase Con Helada:** Concentrada cerca de 1
- La línea punteada en 0.5 es el umbral de decisión
- Distribuciones bien separadas = Buen clasificador

---

## 📈 Gráficos Existentes (Ya presentes en graficos_resultados)

### **curva_roc.png**
Curva ROC del modelo inicial

### **importancia_variables.png**
Importancia de variables/features en la predicción

### **matriz_confusion.png**
Matriz de confusión del modelo inicial

### **xgboost_results.png**
Resultados específicos de XGBoost

---

## 🎯 Resumen de Rendimiento

### Modelo Principal
- **Tipo:** XGBoost
- **Accuracy:** 89.29%
- **Precision:** 90.64%
- **Recall:** 91.15%
- **F1-Score:** 90.89%
- **AUC-ROC:** 96.20%

### Validación Cruzada (5 Folds)
- **F1-Score promedio:** 0.9109 ± 0.0047
- **AUC-ROC promedio:** 0.9742 ± 0.0037
- **Brier Score promedio:** 0.0684 ± 0.0023

### Datos Disponibles
- **Total de registros:** 906
- **Heladas (Clase 1):** 531 (58.6%)
- **Sin heladas (Clase 0):** 375 (41.4%)

---

## 💡 Interpretación General

El modelo demuestra **excelente rendimiento** en la predicción de heladas:

1. ✅ **AUC-ROC muy alto (0.9620)** - Excelente discriminación entre clases
2. ✅ **Precisión y Recall balanceados** - No sacrifica una métrica por otra
3. ✅ **Baja variabilidad en validación cruzada** - Modelo estable y generalizable
4. ✅ **Brier Score bajo (0.0684)** - Error pequeño en probabilidades
5. ✅ **Múltiples modelos evaluados** - Comparación rigurosa

---

## 📁 Ubicación

Todos los gráficos están guardados en:
```
graficos_resultados/
├── 01_curva_roc.png
├── 02_matriz_confusion.png
├── 03_scatter_plot.png
├── 04_precision_recall_curve.png
├── 05_metricas_por_fold.png
├── 06_comparacion_modelos.png
├── 07_roc_comparacion_modelos.png
├── 08_resumen_metricas.png
├── 09_analisis_temporal.png
└── 10_violin_plot.png
```

---

## 🔧 Cómo reproducir

Para regenerar los gráficos en cualquier momento:

```bash
python generate_visualizations.py
```

---

*Documento generado automáticamente - Predicción de Heladas en Puno*
*Fecha: 2026-06-03*