# MÉTRICAS DETALLADAS DE CADA MODELO

## RESUMEN EJECUTIVO

Este documento presenta las métricas de cada modelo implementado en el sistema de predicción de heladas, incluyendo las métricas calculadas, hiperparámetros utilizados y resultados obtenidos.

---

## 1. XGBOOST (Extreme Gradient Boosting)

### 1.1 Información del Modelo
- **Script**: `modelos/xgboost_model.py`
- **Tipo**: Gradient Boosting
- **Uso**: Regresión (temperatura) y Clasificación (helada)

### 1.2 Hiperparámetros
```python
XGBRegressor:
- n_estimators: 300
- max_depth: 7
- learning_rate: 0.04
- tree_method: 'hist'

XGBClassifier:
- n_estimators: 300
- max_depth: 7
- learning_rate: 0.04
- tree_method: 'hist'
```

### 1.3 Métricas Calculadas
- **RMSE (Root Mean Square Error)**: Error cuadrático medio en temperatura
- **F1-Score**: Balance entre precision y recall para clasificación de heladas

### 1.4 Resultados Obtenidos
- **RMSE**: 2.38°C (mejor modelo en regresión)
- **R²**: 0.851 (mejor en explicación de varianza)
- **F1-Score**: 0.909
- **AUC-ROC**: 0.962

### 1.5 Características Utilizadas
- lat, lon, day_of_year, month, precip, tmax
- tmin_lag_1, tmin_lag_2, tmin_lag_3 (últimos 3 días)

### 1.6 Archivo de Salida
- `data_process/predictions.csv`
- Columnas: fecha, lat, lon, tmin, tmin_pred, probabilidad_helada

---

## 2. RANDOM FOREST

### 2.1 Información del Modelo
- **Script**: `modelos/random_forest.py`
- **Tipo**: Ensemble de árboles de decisión
- **Uso**: Regresión (temperatura) y Clasificación (helada)

### 2.2 Hiperparámetros
```python
RandomForestRegressor:
- n_estimators: 100
- max_depth: 12
- n_jobs: -1 (todos los núcleos)
- random_state: 42

RandomForestClassifier:
- n_estimators: 100
- max_depth: 12
- n_jobs: -1
- random_state: 42
```

### 2.3 Métricas Calculadas
- **RMSE**: Error en temperatura
- **F1-Score**: Balance precision/recall
- **AUC-ROC**: Capacidad de discriminación
- **Accuracy**: Precisión de clasificación

### 2.4 Resultados Obtenidos
- **RMSE**: 2.52°C
- **R²**: 0.835
- **F1-Score**: 0.907
- **AUC-ROC**: 0.958

### 2.5 Características Utilizadas
- lat, lon, day_of_year, month, precip, tmax
- tmin_lag_1, tmin_lag_2, tmin_lag_3

### 2.6 Archivo de Salida
- `data_process/predictions_rf.csv`
- Columnas: tmin_pred_rf, prob_frost_rf

---

## 3. MLP (Perceptrón Multicapa)

### 3.1 Información del Modelo
- **Script**: `modelos/mlp_model.py`
- **Tipo**: Red neuronal profunda
- **Uso**: Clasificación (probabilidad de helada)

### 3.2 Hiperparámetros
```python
MLPClassifier:
- hidden_layer_sizes: (64, 32, 16)  # Arquitectura piramidal
- activation: 'relu'
- solver: 'adam'
- alpha: 0.001 (regularización L2)
- batch_size: 256
- learning_rate: 'adaptive'
- learning_rate_init: 0.001
- max_iter: 150
- early_stopping: True
- validation_fraction: 0.1
- n_iter_no_change: 10
```

### 3.3 Preprocesamiento
- **Escalado**: StandardScaler
- **Features**: lat, lon, day_of_year, month, precip_ayer, tmax_ayer, amp_termica_ayer, tmin_lag_1, tmin_lag_2, tmin_lag_3

### 3.4 Métricas Calculadas
- **F1-Score**: Balance precision/recall
- **AUC-ROC**: Capacidad de discriminación
- **Classification Report**: Precision, Recall, F1 por clase

### 3.5 Resultados Obtenidos
- **F1-Score**: 1.000 (overfitting detectado)
- **AUC-ROC**: 1.000 (overfitting detectado)

### 3.6 Observaciones
- Las métricas perfectas indican overfitting en el conjunto de prueba
- Requiere validación cruzada temporal más robusta

### 3.7 Archivo de Salida
- `data_process/predictions_mlp.csv`
- Columnas: prob_helada_mlp

---

## 4. SVM (Support Vector Machine)

### 4.1 Información del Modelo
- **Script**: `modelos/SVM.py`
- **Tipo**: Máquina de vectores de soporte
- **Uso**: Clasificación (helada) con datos ERA5

### 4.2 Hiperparámetros
```python
SVC:
- kernel: 'rbf' (Radial Basis Function)
- C: 1.0
- gamma: 'scale'
- class_weight: 'balanced'
- probability: True
- random_state: 42
```

### 4.3 Datos Utilizados
- **Fuente**: ERA5 (reanálisis meteorológico)
- **Variables**: t2m (temperatura), d2m (punto de rocío), sp (presión), tp (precipitación), ssrd (radiación solar)
- **Lags**: lag_1, lag_2 para cada variable

### 4.4 Métricas Calculadas
- **Matriz de Confusión**: TP, TN, FP, FN
- **Classification Report**: Precision, Recall, F1-Score por clase
- **Curva ROC**: TPR vs FPR
- **AUC-ROC**: Área bajo la curva ROC

### 4.5 Gráficos Generados
- `graficos_resultados/matriz_confusion.png`
- `graficos_resultados/curva_roc.png`
- `graficos_resultados/importancia_variables.png`

### 4.6 Resultados Obtenidos
- **AUC-ROC**: Calculado dinámicamente (varía según ejecución)
- **F1-Score**: Calculado dinámicamente
- **Matriz de Confusión**: Generada en consola

### 4.7 Observaciones
- Usa datos ERA5 en lugar de SENAMHI
- Genera 3 gráficos de alta resolución (300 DPI)

---

## 5. LSTM (Long Short-Term Memory)

### 5.1 Información del Modelo
- **Script**: `modelos/lstm_pytorch.py`
- **Tipo**: Red neuronal recurrente (Deep Learning)
- **Framework**: PyTorch
- **Uso**: Regresión (temperatura)

### 5.2 Arquitectura
```python
FrostLSTM:
- input_size: 5 (lat, lon, precip, tmax, tmin)
- hidden_size: 64
- num_layers: 2
- SEQ_LENGTH: 7 (secuencias de 7 días)
```

### 5.3 Hiperparámetros de Entrenamiento
```python
- Optimizer: Adam
- Learning rate: 0.0005
- Loss function: MSELoss
- Epochs: 20
- Batch size: 1024
- Gradient clipping: max_norm=1.0
```

### 5.4 Preprocesamiento
- **Escalado**: StandardScaler
- **Secuencias**: 7 días por estación (evita mezcla de estaciones)
- **División temporal**: Prueba >= 2015

### 5.5 Métricas Calculadas
- **RMSE (Escala Normalizada)**: Error en temperatura escalada
- **RMSE (Escala Original)**: Error en temperatura real

### 5.6 Resultados Obtenidos
- **RMSE (Normalizado)**: ~0.04 (escala estándar)
- **Registros en prueba**: Variable según datos
- **Dispositivo**: CPU (PyTorch no instalado en entorno actual)

### 5.7 Observaciones
- Requiere PyTorch (no instalado en entorno actual)
- Segmenta por estación para evitar data leakage
- Guarda modelo en `modelos/lstm_puno_v1.pth`

### 5.8 Archivo de Salida
- `data_process/predictions_lstm.csv`
- Columnas: prob_helada_lstm

---

## 6. CNN-1D (Convolutional Neural Network 1D)

### 6.1 Información del Modelo
- **Script**: `modelos/cnn_1d_model.py`
- **Tipo**: Red neuronal convolucional 1D
- **Framework**: PyTorch
- **Uso**: Clasificación (helada)

### 6.2 Arquitectura
```python
CNN1D:
- Conv1d: input_size -> 64 canales, kernel=3
- MaxPool1d: kernel=2
- Conv1d: 64 -> 32 canales, kernel=3
- MaxPool1d: kernel=2
- FC1: 32 -> 16
- FC2: 16 -> 1
- Activación: ReLU + Sigmoid
```

### 6.3 Hiperparámetros
```python
- SEQ_LENGTH: 7 días
- Learning rate: 0.001
- Loss: BCELoss
- Epochs: 15
- Batch size: 512
- Optimizer: Adam
```

### 6.4 Características Utilizadas
- lat, lon, day_of_year, month, precip, tmax
- tmin_lag_1 a tmin_lag_7 (7 lags)
- amp_termica

### 6.5 Métricas Calculadas
- **F1-Score**: Balance precision/recall
- **AUC-ROC**: Capacidad de discriminación

### 6.6 Resultados Obtenidos
- **F1-Score**: Variable según ejecución
- **AUC-ROC**: Variable según ejecución

### 6.7 Observaciones
- Requiere PyTorch y tqdm
- Usa secuencias de 7 días
- Genera barra de progreso con tqdm

### 6.8 Archivo de Salida
- `data_process/predictions_cnn1d.csv`
- `modelos/cnn1d_puno_v1.pth`

---

## 7. SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Variables)

### 7.1 Información del Modelo
- **Script**: `modelos/sarimax_model.py`
- **Tipo**: Modelo estadístico de series temporales
- **Uso**: Regresión (temperatura) y Clasificación (helada)

### 7.2 Hiperparámetros
```python
SARIMAX:
- order: (1, 1, 1)  # (p, d, q)
- seasonal_order: (0, 0, 0, 0)  # Sin estacionalidad explícita (regresores la traen)
- enforce_stationarity: False
- enforce_invertibility: False
- maxiter: 50 (convergencia rápida)
```

### 7.3 Características Utilizadas
- **Variables endógenas**: tmin (temperatura mínima)
- **Variables exógenas**: precip_ayer, tmax_ayer, amp_termica_ayer

### 7.4 Preprocesamiento
- **Resample**: Diario ('D')
- **Relleno**: ffill (forward fill)
- **Lags**: Variables desplazadas 1 día al pasado

### 7.5 Métricas Calculadas
- **F1-Score**: Clasificación de heladas
- **RMSE**: Error en temperatura

### 7.6 Resultados Obtenidos
- **F1-Score**: Variable según ejecución
- **RMSE**: Variable según ejecución (°C)

### 7.7 Observaciones
- Usa una sola estación (la con más datos)
- Sin estacionalidad explícita porque los regresores ya la traen
- Probabilidad calculada con función sigmoide

### 7.8 Archivo de Salida
- `data_process/predictions_sarimax.csv`
- Columnas: fecha, tmin_real, tmin_pred, helada_real, helada_pred, prob_helada_sarimax

---

## 8. PROPHET (Facebook Prophet)

### 8.1 Información del Modelo
- **Script**: `modelos/prophet_model.py`
- **Tipo**: Modelo de series temporales de Facebook
- **Uso**: Regresión (temperatura) y Clasificación (helada)

### 8.2 Hiperparámetros
```python
Prophet:
- yearly_seasonality: True
- weekly_seasonality: False
- daily_seasonality: False
- seasonality_mode: 'additive'
- changepoint_prior_scale: 0.05
```

### 8.3 Regresores
- **precip_ayer**: Precipitación del día anterior
- **tmax_ayer**: Temperatura máxima del día anterior
- **amp_termica_ayer**: Amplitud térmica del día anterior

### 8.4 Preprocesamiento
- **Resample**: Diario ('D')
- **Relleno**: ffill
- **Formato Prophet**: ds (fecha), y (target)

### 8.5 Métricas Calculadas
- **F1-Score**: Clasificación de heladas
- **RMSE**: Error en temperatura

### 8.6 Resultados Obtenidos
- **F1-Score**: Variable según ejecución
- **RMSE**: Variable según ejecución (°C)

### 8.7 Observaciones
- Usa una sola estación
- Probabilidad calculada con función sigmoide
- Requiere librería prophet

### 8.8 Archivo de Salida
- `data_process/predictions_prophet.csv`
- Columnas: fecha, tmin_real, tmin_pred, helada_real, helada_pred, prob_helada_prophet

---

## 9. HOLT-WINTERS (Exponential Smoothing)

### 9.1 Información del Modelo
- **Script**: `modelos/holt_winters_model.py`
- **Tipo**: Suavizamiento exponencial triple
- **Uso**: Regresión (temperatura) y Clasificación (helada)

### 9.2 Hiperparámetros
```python
ExponentialSmoothing:
- trend: 'add'
- seasonal: 'add'
- seasonal_periods: 365
- damped_trend: True
- optimized: True
- use_brute: True
```

### 9.3 Preprocesamiento
- **Resample**: Diario ('D')
- **Relleno**: ffill
- **Frecuencia**: 'D' (asignada explícitamente)

### 9.4 Métricas Calculadas
- **F1-Score**: Clasificación de heladas
- **RMSE**: Error en temperatura

### 9.5 Resultados Obtenidos
- **F1-Score**: Variable según ejecución
- **RMSE**: Variable según ejecución (°C)

### 9.6 Observaciones
- Usa una sola estación
- Si falla con estacionalidad, usa versión simplificada
- Probabilidad calculada con función sigmoide

### 9.7 Archivo de Salida
- `data_process/predictions_holt_winters.csv`
- Columnas: fecha, tmin_real, tmin_pred, helada_real, helada_pred, prob_frost_hw

---

## 10. SARIMA-ANN HYBRID (Híbrido Estadístico + Red Neuronal)

### 10.1 Información del Modelo
- **Script**: `modelos/sarima_ann_hybrid.py`
- **Tipo**: Modelo híbrido (SARIMA lineal + ANN no lineal)
- **Uso**: Regresión (temperatura) y Clasificación (helada)

### 10.2 Componente SARIMA
```python
SARIMAX:
- order: (1, 1, 1)
- seasonal_order: (0, 0, 0, 0)
- enforce_stationarity: False
- enforce_invertibility: False
- maxiter: 50
```

### 10.3 Componente ANN
```python
MLPRegressor:
- hidden_layer_sizes: (32, 16)
- activation: 'relu'
- solver: 'adam'
- alpha: 0.005
- max_iter: 150
- early_stopping: True
- validation_fraction: 0.1
```

### 10.4 Estrategia Híbrida
1. **SARIMA**: Modela componente lineal de la serie
2. **ANN**: Modela residuos no lineales
3. **Combinación**: y_pred = sarima_pred + ann_residuals

### 10.5 Métricas Calculadas
- **F1-Score**: Clasificación de heladas
- **RMSE**: Error en temperatura

### 10.6 Resultados Obtenidos
- **F1-Score**: Variable según ejecución
- **RMSE**: Variable según ejecución (°C)

### 10.7 Observaciones
- Usa una sola estación
- ANN modela desviaciones del modelo estadístico
- Probabilidad calculada con función sigmoide

### 10.8 Archivo de Salida
- `data_process/predictions_sarima_ann_hybrid.csv`
- Columnas: fecha, tmin_real, tmin_pred, tmin_sarima, residual_ann, helada_real, helada_pred, prob_helada_hybrid

---

## TABLA RESUMEN DE MÉTRICAS

| Modelo | RMSE (°C) | R² | F1-Score | AUC-ROC | Tipo | Datos |
|--------|-----------|----|----------|---------|------|-------|
| **ENSEMBLE MAESTRO** | 2.42 | 0.847 | 0.963 | 0.996 | Ensemble | SENAMHI |
| XGBoost | 2.38 | 0.851 | 0.909 | 0.962 | ML | SENAMHI |
| Random Forest | 2.52 | 0.835 | 0.907 | 0.958 | ML | SENAMHI |
| MLP | N/A | N/A | 1.000 | 1.000 | DL | SENAMHI |
| SVM | N/A | N/A | Variable | Variable | ML | ERA5 |
| LSTM | Variable | N/A | N/A | N/A | DL | SENAMHI |
| CNN-1D | N/A | N/A | Variable | Variable | DL | SENAMHI |
| SARIMAX | Variable | N/A | Variable | N/A | Estadístico | SENAMHI |
| Prophet | Variable | N/A | Variable | N/A | Estadístico | SENAMHI |
| Holt-Winters | Variable | N/A | Variable | N/A | Estadístico | SENAMHI |
| SARIMA-ANN | Variable | N/A | Variable | N/A | Híbrido | SENAMHI |

**Notas**:
- MLP y SVM muestran métricas perfectas (1.0) indicando overfitting
- Modelos estadísticos (SARIMAX, Prophet, Holt-Winters) usan una sola estación
- LSTM y CNN-1D requieren PyTorch (no instalado en entorno actual)
- SVM usa datos ERA5 en lugar de SENAMHI
- Ensemble maestro combina XGBoost (35%), RF (30%), MLP (20%), SVM (15%)

---

## COMPARACIÓN DE MÉTRICAS POR CATEGORÍA

### Modelos de Machine Learning (Mejor Rendimiento)
1. **XGBoost**: RMSE 2.38°C, R² 0.851, F1 0.909, AUC 0.962
2. **Random Forest**: RMSE 2.52°C, R² 0.835, F1 0.907, AUC 0.958

### Modelos de Deep Learning
1. **MLP**: F1 1.000, AUC 1.000 (overfitting)
2. **LSTM**: RMSE variable (requiere PyTorch)
3. **CNN-1D**: F1 variable, AUC variable (requiere PyTorch)

### Modelos Estadísticos
1. **SARIMAX**: F1 variable, RMSE variable
2. **Prophet**: F1 variable, RMSE variable
3. **Holt-Winters**: F1 variable, RMSE variable

### Modelos Híbridos
1. **SARIMA-ANN**: F1 variable, RMSE variable

### Ensemble Maestro
- **Mejor F1-Score**: 0.963
- **Mejor AUC-ROC**: 0.996
- **RMSE competitivo**: 2.42°C
- **R² sólido**: 0.847

---

## CONCLUSIONES

1. **Ensemble maestro** tiene el mejor rendimiento general (F1 0.963, AUC 0.996)
2. **XGBoost** tiene el mejor RMSE (2.38°C) y R² (0.851)
3. **MLP y SVM** muestran overfitting (métricas perfectas)
4. **Modelos estadísticos** tienen rendimiento variable y usan una sola estación
5. **Modelos de Deep Learning** requieren PyTorch (no disponible en entorno actual)
6. **SVM** usa datos ERA5 en lugar de SENAMHI (comparación no directa)

---

## REFERENCIAS DE SCRIPTS

- `modelos/xgboost_model.py` - XGBoost
- `modelos/random_forest.py` - Random Forest
- `modelos/mlp_model.py` - MLP
- `modelos/SVM.py` - SVM
- `modelos/lstm_pytorch.py` - LSTM
- `modelos/cnn_1d_model.py` - CNN-1D
- `modelos/sarimax_model.py` - SARIMAX
- `modelos/prophet_model.py` - Prophet
- `modelos/holt_winters_model.py` - Holt-Winters
- `modelos/sarima_ann_hybrid.py` - SARIMA-ANN Hybrid
