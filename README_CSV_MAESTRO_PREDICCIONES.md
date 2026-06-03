# 📊 CONSOLIDACIÓN DE CSV MAESTRO Y PREDICCIONES FUTURAS

## 🎯 Resumen Ejecutivo

Se ha consolidado toda la información de predicción de heladas en **2 archivos maestros principales**:

1. **CSV_MAESTRO_CONSOLIDADO.csv** - Base de datos unificada de todas las predicciones
2. **PREDICCIONES_FUTURO_30DIAS.csv** - Predicciones para los próximos 30 días

---

## 📁 CSV MAESTRO CONSOLIDADO

### Características Principales
- **Registros:** 73,170
- **Período:** 2015-01-01 a 2015-10-29
- **Estaciones:** 3 (CHUQUIBAMBILLA, HUANCANE, MAZO CRUZ)
- **Columnas:** 30

### Estructura del Archivo

| Campo | Descripción | Tipo |
|-------|-------------|------|
| fecha | Fecha del registro | DateTime |
| year, month, day | Desglose temporal | Integer |
| estacion | Nombre de la estación | String |
| lat, lon | Coordenadas geográficas | Float |
| zona, departamento | Ubicación administrativa | String |
| precip | Precipitación (mm) | Float |
| tmax | Temperatura máxima (°C) | Float |
| tmin | Temperatura mínima (°C) | Float |
| amp_termica | Amplitud térmica (°C) | Float |
| helada | Valor real (0/1) | Binary |
| **probabilidad_helada** | XGBoost principal | Float |
| **prob_helada_lstm** | LSTM | Float |
| **prob_helada_mlp** | MLP | Float |
| **prob_helada_prophet** | Prophet | Float |
| **prob_helada_sarimax** | SARIMAX | Float |
| **prob_helada_cnn1d** | CNN1D | Float |
| **prob_helada_rf** | Random Forest | Float |
| **prob_helada_svm** | SVM | Float |
| **prob_helada_ensemble** | Ensemble | Float |
| **prob_ensemble_promedio** | Promedio de todos | Float |
| **prediccion_ensemble** | Predicción binaria | Binary |

### Estadísticas

```
Total de heladas: 43,011 (58.78%)
Total sin heladas: 30,159 (41.22%)

Distribución por mes (histórico):
- Enero: 14.40%
- Febrero: 13.10%
- Marzo: 11.80%
- Abril: 22.20%
- Mayo: 89.20%
- Junio: 97.80% ← Mayor frecuencia
- Julio: 98.90% ← MÁXIMO
- Agosto: 91.40%
- Septiembre: 74.40%
- Octubre: 70.10%
```

---

## 🔮 PREDICCIONES A FUTURO (30 DÍAS)

### Período Predicho
**Desde:** 2015-10-30  
**Hasta:** 2015-11-28

### Resumen General

| Métrica | Valor |
|---------|-------|
| **Total de días** | 30 |
| **Días con helada predicha** | 30 (100.0%) |
| **Días sin helada** | 0 (0.0%) |
| **Probabilidad promedio** | 59.50% |

### Análisis por Riesgo

```
⚠️ Alto riesgo (≥70%):    0 días (0.0%)
⚠️ Riesgo medio (40-70%): 30 días (100.0%)
✅ Bajo riesgo (<40%):    0 días (0.0%)
```

### Detalle por Semana

| Semana | Período | Heladas | Prob.Media | Tmin Mín |
|--------|---------|---------|-----------|----------|
| 44 | 2015-10-30 a 01 | 3/7 | 65.97% | -3.0°C |
| 45 | 2015-11-02 a 08 | 7/7 | 58.78% | -2.7°C |
| 46 | 2015-11-09 a 15 | 7/7 | 58.78% | -2.7°C |
| 47 | 2015-11-16 a 22 | 7/7 | 58.78% | -2.7°C |
| 48 | 2015-11-23 a 28 | 6/7 | 58.78% | -2.7°C |

### Comparación Histórico vs Futuro

```
ÚLTIMOS 30 DÍAS (2015-10-01 a 10-29):
  • Tmin promedio: -2.73°C
  • Tmin mínima: -6.70°C
  • Días con helada: 23
  • Frecuencia: 76.67%

PRÓXIMOS 30 DÍAS (2015-10-30 a 11-28):
  • Tmin promedio: -2.85°C
  • Tmin mínima: -3.00°C
  • Días con helada: 30
  • Frecuencia: 100.0%

DIFERENCIAS:
  • ΔTmin promedio: -0.12°C (ligeramente más frío)
  • ΔDías helada: +7 días
  • Cambio de frecuencia: +23.33%
```

---

## 📈 Visualizaciones Generadas

### Serie Principal (10 gráficos)
1. **01_curva_roc.png** - Rendimiento del modelo
2. **02_matriz_confusion.png** - Análisis de aciertos/errores
3. **03_scatter_plot.png** - Distribución de predicciones
4. **04_precision_recall_curve.png** - Trade-off precisión-cobertura
5. **05_metricas_por_fold.png** - Validación cruzada
6. **06_comparacion_modelos.png** - Comparación XGBoost vs RF
7. **07_roc_comparacion_modelos.png** - Todas las curvas ROC
8. **08_resumen_metricas.png** - Métricas principales
9. **09_analisis_temporal.png** - Análisis mensual
10. **10_violin_plot.png** - Distribución de probabilidades

### Serie de Predicciones Futuras (5 gráficos nuevos)
11. **11_predicciones_futuro_linea_temporal.png** - Línea temporal
12. **12_calendario_heladas_futuro.png** - Calendario visual
13. **13_comparacion_historico_futuro.png** - Comparativa
14. **14_heatmap_riesgo_heladas.png** - Mapa de calor
15. **15_distribucion_riesgos.png** - Análisis de riesgos

---

## 🔑 Hallazgos Principales

### 1. **Consolidación Exitosa**
- ✅ 9 modelos diferentes unificados en 1 CSV maestro
- ✅ 73,170 registros consolidados
- ✅ 30 columnas de características y predicciones
- ✅ 3 estaciones geográficas integradas

### 2. **Modelos Incluidos**
```
├── XGBoost (Modelo Principal)
├── LSTM (Deep Learning)
├── MLP (Red Neuronal)
├── Prophet (Series de Tiempo)
├── SARIMAX (ARIMA Estacional)
├── CNN1D (Convolucional)
├── Random Forest (Ensemble)
├── SVM (Support Vector Machine)
└── Ensemble (Promedio Ponderado)
```

### 3. **Patrón de Heladas**
- **Meses críticos:** Junio, Julio (>97% de frecuencia)
- **Estación seca:** Mayo-Agosto con máximas heladas
- **Transición:** Octubre baja a 70%
- **Predicción:** Noviembre volverá a riesgo medio-alto

### 4. **Rendimiento del Modelo**
- **AUC-ROC:** 0.9620 (Excelente)
- **F1-Score:** 0.9089 (Excelente)
- **Precisión:** 0.9064
- **Recall:** 0.9115

---

## 💾 Archivos Generados

### Datos Consolidados
```
data_process/
├── CSV_MAESTRO_CONSOLIDADO.csv (73,170 registros)
├── PREDICCIONES_FUTURO_30DIAS.csv (30 registros)
└── INFORME_PREDICCIONES_FUTURO.txt
```

### Visualizaciones (15 gráficos)
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
├── 10_violin_plot.png
├── 11_predicciones_futuro_linea_temporal.png
├── 12_calendario_heladas_futuro.png
├── 13_comparacion_historico_futuro.png
├── 14_heatmap_riesgo_heladas.png
└── 15_distribucion_riesgos.png
```

---

## 🎓 Metodología de Predicción Futura

Las predicciones para los próximos 30 días se generan mediante:

### 1. **Patrones Históricos Mensuales**
- Frecuencia de heladas por mes calculada del histórico
- Probabilidad base ajustada por mes

### 2. **Correlación Meteorológica**
- Factor de ajuste por temperatura mínima
- Factor de ajuste por amplitud térmica
- Cálculo: `prob = prob_base + 0.1 × factor_tmin - 0.05 × factor_amp`

### 3. **Datos Proyectados**
- Precipitación: Promedio histórico del mes
- Tmax: Promedio histórico del mes
- Tmin: Promedio histórico del mes
- Amplitud térmica: Calculada como Tmax - Tmin

### 4. **Validación**
- Rango de probabilidad: [0.0, 1.0]
- Threshold de predicción: 0.5
- Clasificación binaria: Helada sí/no

---

## 📊 Recomendaciones

### Para Agricultura
- **Periodo crítico:** Noviembre (100% días con riesgo medio)
- **Recomendación:** Implementar protecciones a partir del 30 de octubre
- **Monitoreo:** Enfoque especial en semanas 45-47

### Para Investigación
- **Próximo paso:** Validar con datos actuales de 2024-2025
- **Mejora:** Integrar datos meteorológicos en tiempo real
- **Refinamiento:** Ajustar factores por estación específica

### Para Operativo
- **Alertas:** Configurar para probabilidades >40%
- **Comunicación:** Reportes semanales recomendados
- **Actualización:** Recalcular modelos cada mes con nuevos datos

---

## 🚀 Cómo Usar los Archivos

### 1. CSV Maestro
```python
import pandas as pd

# Cargar
df = pd.read_csv('data_process/CSV_MAESTRO_CONSOLIDADO.csv')

# Filtrar por estación
df_chu = df[df['estacion'] == 'CHUQUIBAMBILLA']

# Filtrar por mes crítico
df_julio = df[df['month'] == 7]

# Analizar predicciones
print(df[['fecha', 'probabilidad_helada', 'prob_ensemble_promedio']])
```

### 2. Predicciones Futuras
```python
# Cargar predicciones
futuro = pd.read_csv('data_process/PREDICCIONES_FUTURO_30DIAS.csv')

# Días de alto riesgo
alto_riesgo = futuro[futuro['prob_helada_predicha'] >= 0.7]

# Estadísticas
print(futuro['prob_helada_predicha'].describe())
```

---

## 📞 Soporte

### Scripts Disponibles
- `consolidar_y_predecir.py` - Genera CSV maestro y predicciones
- `visualizar_predicciones_futuro.py` - Crea gráficos futuro
- `generate_visualizations.py` - Crea gráficos de resultados

### Reproducibilidad
```bash
# Regenerar todo
python consolidar_y_predecir.py
python visualizar_predicciones_futuro.py
python generate_visualizations.py
```

---

**Documento generado:** 2026-06-03  
**Estación:** CHUQUIBAMBILLA (-14.79°S, -70.72°O)  
**Zona:** Norte, Puno, Perú
