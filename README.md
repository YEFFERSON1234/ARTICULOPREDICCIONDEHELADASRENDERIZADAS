# Predicción de Heladas Renderizadas en Puno

Este repositorio contiene un proyecto de predicción de heladas para la región de Puno, Perú. El objetivo es usar datos meteorológicos y modelos de machine learning para predecir la temperatura mínima y el riesgo de helada, además de preparar resultados para visualización geoespacial.

## 📁 Estructura principal

- `limpiezadedatos/`
  - `datos_heladas_puno_REAL.csv` — datos históricos de temperatura y precipitación para estaciones en Puno.
  - `predictions.csv` — resultados de predicción generados por el modelo XGBoost.
  - `era5/descargardataset.py` — script para descargar datos ERA5.
- `modelos/`
  - `xgboost_model.py` — modelo principal que genera predicción de `tmin` y probabilidad de helada.
  - `randomforest.py` — modelo de bosque aleatorio.
  - `LSTMpythorch.py` — modelo LSTM en PyTorch.
  - `ensamble.py` — punto de ensamblaje o combinación de modelos.
- `Archivos.tiff-renderizar/` — archivos de terreno y DEM para renderizado.
- `Render/` — gráficos y visualizaciones de salida.
- `graficos_resultados/` — resultados gráficos ya generados.
- `README.md` — documentación del proyecto.
- `requeriment.txt` — dependencias recomendadas.

## 🎯 ¿Qué predice este proyecto?

El proyecto predice:

- `tmin_pred`: temperatura mínima diaria estimada.
- `probabilidad_helada`: probabilidad de que ocurra una helada.

Además, el archivo `predictions.csv` contiene estos campos junto a las variables observadas de entrada.

## 🧠 Variables usadas en el modelo XGBoost

El script `modelos/xgboost_model.py` usa estas columnas como características:

- `lat`, `lon` — ubicación de la estación.
- `day_of_year`, `month`, `year` — variables temporales.
- `precip` — precipitación diaria.
- `tmax` — temperatura máxima.
- `tmin_lag_1`, `tmin_lag_2`, `tmin_lag_3` — temperaturas mínimas de los 3 días anteriores.

Y predice:

- `tmin_pred` — temperatura mínima estimada.
- `probabilidad_helada` — probabilidad de helada.

## 📌 Columnas de `limpiezadedatos/predictions.csv`

- `year`, `month`, `day`: fecha desglosada.
- `precip`: precipitación diaria.
- `tmax`: temperatura máxima observada.
- `tmin`: temperatura mínima observada.
- `estacion`: nombre de la estación meteorológica.
- `lat`, `lon`: coordenadas geográficas.
- `zona`: subdivisión dentro de Puno.
- `departamento`: departamento político (`PUNO`).
- `fecha`: fecha completa.
- `frost`: indicador real de helada (1 = tmin ≤ 0°C, 0 = no helada).
- `day_of_year`: día del año.
- `tmin_lag_1`, `tmin_lag_2`, `tmin_lag_3`: tmin de los 3 días anteriores.
- `tmin_pred`: temperatura mínima predicha por el modelo.
- `probabilidad_helada`: probabilidad estimada de helada.

## 🛠️ Cómo instalar dependencias

Se recomienda usar Python 3.11.9 y crear un entorno virtual.

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==2.4.3 pandas==3.0.2 scipy==1.17.1 xgboost==3.2.0 scikit-learn==1.8.0
pip install rasterio==1.4.4 affine==2.4.0 click==8.3.3 cligj==0.7.2 click-plugins==1.1.1.2
pip install matplotlib==3.10.9 seaborn==0.13.2
pip install opencv-python==4.13.0.92 pillow==12.1.1
pip install pyopengl glfw joblib==1.5.3 threadpoolctl==3.6.0 networkx==3.6.1 sympy==1.13.1 jinja2==3.1.6 fsspec==2026.2.0
```

## ▶️ Cómo ejecutar la predicción principal

Desde la carpeta raíz del proyecto:

```powershell
python modelos/xgboost_model.py
```

Esto generará o actualizará el archivo:

- `limpiezadedatos/predictions.csv`

## 🌐 Cómo usar datos ERA5

El script `limpiezadedatos/era5/descargardataset.py` está preparado para descargar datos ERA5 de Copernicus. Debes:

1. Tener una cuenta en Copernicus Climate Data Store.
2. Configurar tu `cdsapi` con la clave de acceso.
3. Ejecutar el script para descargar las variables deseadas.

Este repositorio puede integrar ERA5 para mejorar las predicciones con más variables, como:

- `2m_temperature`
- `2m_dewpoint_temperature`
- `surface_pressure`
- `total_precipitation`
- `surface_solar_radiation_downwards`

## 📈 Recomendaciones para mejorar el proyecto

- Agregar más variables de ERA5 (humedad, presión, radiación) al dataset de entrenamiento.
- Unificar los datos de estaciones con las variables climáticas ERA5.
- Entrenar con más años y usar validación cruzada temporal.
- Añadir métricas de evaluación: RMSE, MAE, precisión de helada, matriz de confusión.
- Generar gráficos de `tmin` real vs predicho y curva ROC para el modelo de helada.

## 📝 Resultados actuales

El proyecto ya tiene un flujo funcional donde se generan predicciones diarias para la temperatura mínima y el riesgo de helada. La predicción se basa en datos históricos de estaciones de Puno y usa una combinación de modelos en `modelos/`.

## 📚 Archivos útiles

- `arquitectura.txt` — explica el diagrama de entrenamiento y renderizado.
- `arbol_completo.txt` — lista completa de carpetas y archivos.
- `graficos_resultados/` — contiene salidas visuales y reportes.

## ✅ Conclusión

Este repositorio es una base sólida para una predicción de heladas en Puno, con datos reales, modelos de ML y visualizaciones 3D. El siguiente paso es enriquecer los datos con ERA5 y documentar los pasos de limpieza y evaluación en un script o notebook adicional.
