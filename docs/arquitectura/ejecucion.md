# 🚀 Guía de Ejecución del Proyecto

Paso a paso cómo ejecutar el proyecto de predicción de heladas en Puno y ver los resultados.

---

## 📋 Requisitos Previos

- [x] Python 3.11.9 instalado
- [x] Dependencias instaladas (ver [setup.md](../configuracion/setup.md))
- [x] Datos históricos en `data_process/`
- [x] Modelos en `modelos/`

---

## 🎯 Ejecución Rápida (Recomendado)

### **Opción 1: Pipeline Automatizado**

```powershell
# Ejecutar todo de una vez
python main_pipeline.py
```

Esto ejecuta automáticamente:
1. ✅ Preparación de datos
2. ✅ Entrenamiento de modelos
3. ✅ Predicción
4. ✅ Preparación del terreno 3D
5. ✅ Visualización

### **Opción 2: Con Opciones Específicas**

```powershell
# Verbose (con detalles)
python main_pipeline.py --verbose

# Solo predicción
python main_pipeline.py --predict-only

# Solo visualización
python main_pipeline.py --visualize-only

# Saltar pasos específicos
python main_pipeline.py --skip 1 2
```

### **Opción 3: Verificación Rápida**

```powershell
# Verificar instalación antes de ejecutar
python quickstart.py
```

---

## 📊 Flujo Detallado Paso a Paso

### **Paso 1: Procesar datos ERA5 (Opcional)**

Si tienes archivos `.nc` de ERA5 en `data/datos_era5_puno/`:

```powershell
python utils/process_csv.py
```

**Genera**: `data/era5_procesado_maestro.csv`

---

### **Paso 2: Unificar datos SENAMHI + ERA5**

Abre el notebook Jupyter:

```powershell
jupyter notebook unificar_datos.ipynb
```

**Pasos en el notebook**:
1. Cargar datos de SENAMHI (`data_process/datos_heladas_puno_REAL.csv`)
2. Cargar datos de ERA5 procesados
3. Unificar por fecha y coordenadas
4. Imputar valores faltantes
5. Guardar resultado

**Genera**: `data_process/dataset_ML_final_completo.csv`

---

### **Paso 3: Entrenar Modelos**

#### Opción A: Usar el script de entrenamiento

```powershell
python src/train.py
```

#### Opción B: Entrenar modelos individuales

```powershell
# XGBoost (modelo principal)
python modelos/xgboost_model.py

# Random Forest
python modelos/random_forest.py

# LSTM
python modelos/lstm_pytorch.py

# Ver docs/datos-modelos/ para todos los modelos
```

**Genera**: Modelos entrenados en `modelos/*.pth` y `modelos/*.pkl`

---

### **Paso 4: Generar Predicciones**

```powershell
python scripts/consolidar_y_predecir.py
```

O con el pipeline maestro:

```powershell
python main_pipeline.py --predict-only
```

**Genera**: 
- `data_process/predictions.csv`
- `data_process/predictions_ensemble.csv`
- `data_process/PREDICCIONES_FUTURO_30DIAS.csv`

---

### **Paso 5: Preparar Terreno 3D (DEM)**

```powershell
python Render/Mapa/convert_tiles_to_csv.py
```

Esto procesa los archivos GeoTIFF en `Archivos.tiff-renderizar/` y los convierte a CSV comprimido.

**Genera**: `Render/Mapa/dem_puno_render.csv.gz` (5.9 MB)

---

### **Paso 6: Visualizar Resultados**

#### **Opción A: Visualizador Unificado (RECOMENDADO)**

```powershell
python visualization/unified_visualizer.py
```

**Modos disponibles**:

```powershell
# Interactivo 3D (requiere OpenGL)
python visualization/unified_visualizer.py --mode interactive

# Estático (siempre funciona)
python visualization/unified_visualizer.py --mode static

# Animación temporal
python visualization/unified_visualizer.py --mode animated

# Mapa de riesgo de helada
python visualization/unified_visualizer.py --mode risk
```

**Controles del visualizador 3D**:
- 🖱️ **Mouse arrastrar**: Rotar vista
- 🔍 **Scroll / +/-**: Zoom
- ⬅️⬜⬆️: Mover cámara
- 🔄 **Espacio**: Auto-rotación
- 🔁 **R**: Reset cámara
- ❌ **ESC**: Salir

#### **Opción B: Visualizadores Específicos**

```powershell
# Visualizador principal (OpenGL)
python visualization/main.py

# Visualizador estático
python Render/Mapa/draw_from_csv.py

# Visualizador con riesgo de helada
python Render/Mapa/visualizer_with_risk.py

# Visualizador animado
python Render/Mapa/animated_visualizer.py
```

---

## 📈 Generación de Gráficos y Reportes

### **Gráficos Comparativos**

```powershell
python scripts/graficar_metricas_comparativas.py
```

Genera gráficos de comparación entre modelos.

### **Informe General**

```powershell
python scripts/generar_informe_general.py
```

Genera reportes en texto/markdown.

### **Gráficos de Predicción**

```powershell
python scripts/graficar_prediccion_2026_06_03.py
```

Genera visualizaciones de predicciones específicas.

---

## 🧪 Tests (Opcional)

```powershell
# Todos los tests
python -m pytest tests/ -v

# Solo tests de datos
python -m pytest tests/test_data_preparation.py -v

# Tests específicos
python -m pytest tests/test_data_preparation.py::TestDataIntegrity::test_data_integrity -v
```

---

## 📊 Variables Disponibles

### **De entrada (features)**
- `lat`, `lon` - Coordenadas
- `day_of_year`, `month`, `year` - Temporales
- `precip`, `tmax`, `tmin` - Meteorológicas
- `tmin_lag_1`, `tmin_lag_2`, `tmin_lag_3` - Históricos

### **De salida (predictions)**
- `tmin_pred` - Temperatura mínima predicha (°C)
- `probabilidad_helada` - Probabilidad (0-1)
- `helada` - Clasificación (0/1)

---

## 💾 Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `data_process/predictions.csv` | Predicciones individuales |
| `data_process/predictions_ensemble.csv` | Predicciones ensemble |
| `data_process/PREDICCIONES_FUTURO_30DIAS.csv` | Futuro 30 días |
| `Render/Mapa/dem_puno_render.csv.gz` | DEM procesado |
| `graficos_resultados/` | Gráficos generados |

---

## 🔧 Troubleshooting

### **Error: "Module not found"**
```powershell
pip install -r requirements.txt --upgrade
```

### **Error: "No data files found"**
```powershell
# Verifica que existan:
ls data_process/datos_heladas_puno_REAL.csv
ls data_process/dataset_ML_final_completo.csv
```

### **Error: "OpenGL not available"**
```powershell
# Usa modo estático
python visualization/unified_visualizer.py --mode static
```

### **Error: "CUDA/GPU not available"**
```powershell
# Esto es normal, el sistema usará CPU automáticamente
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 📞 Más Información

- 📚 [Arquitectura del Sistema](README.md)
- 📊 [Resultados y Métricas](../datos-modelos/)
- ⚙️ [Configuración](../configuracion/)
- 🔧 [Mejoras Implementadas](../mantenimiento/IMPROVEMENTS.md)

---

**Última actualización:** 2026-06-07