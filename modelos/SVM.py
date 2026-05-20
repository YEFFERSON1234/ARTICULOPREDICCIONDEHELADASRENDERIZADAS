import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# 1. PASO 1: CARGAR ARCHIVOS .NC DESDE LA RUTA ABSOLUTA
# =====================================================================
print("-> Iniciando el sistema de predicción de heladas...")
print("-> Buscando archivos NetCDF en la ruta del Escritorio...")

# Usamos la ruta absoluta exacta de tu computadora para evitar fallos de dirección
carpeta_datos = r".."
archivos = glob.glob(os.path.join(carpeta_datos, "*.nc"))

if not archivos:
    raise FileNotFoundError(
        f"\n[!] ERROR: No se encontraron archivos .nc en la ruta especificada:\n"
        f"    {carpeta_datos}\n"
        f"    Por favor, verifica que la carpeta exista en tu Escritorio y contenga los datos."
    )

print(f"[OK] Se encontraron {len(archivos)} archivos mensuales para procesar.")
print("-> Leyendo datos multidimensionales (unificando meses)...")

# 'join=override' y 'compat=override' ignoran diferencias decimales microscópicas de las coordenadas
ds = xr.open_mfdataset(archivos, combine='by_coords', coords='minimal', compat='override', join='override')

# Convertimos las temperaturas de Kelvin a Celsius de inmediato
ds['t2m'] = ds['t2m'] - 273.15
ds['d2m'] = ds['d2m'] - 273.15

print("-> Calculando promedio para la región de Puno...")
# Promediamos espacialmente la cuadrícula para consolidar las provincias en una línea de tiempo
ds_promedio = ds.mean(dim=['latitude', 'longitude'])

# Limpiamos remanentes geográficos para que Pandas no genere conflictos de índices
if 'latitude' in ds_promedio.coords:
    ds_promedio = ds_promedio.drop_vars('latitude')
if 'longitude' in ds_promedio.coords:
    ds_promedio = ds_promedio.drop_vars('longitude')

# Convertimos el cubo de datos a una tabla plana (DataFrame)
df = ds_promedio.to_dataframe().reset_index()
df = df.sort_values('time').reset_index(drop=True)


# =====================================================================
# 2. PASO 2: INGENIERÍA DE CARACTERÍSTICAS (PREPARACIÓN PARA PREVISIÓN)
# =====================================================================
print("-> Creando desfases temporales (Lags) para predecir el futuro...")

# Creamos la etiqueta de helada: 1 si la temperatura es <= 0°C, 0 si no.
df['es_helada'] = (df['t2m'] <= 0).astype(int)

# Variables meteorológicas que el modelo analizará
variables_clave = ['t2m', 'd2m', 'sp', 'tp', 'ssrd']

# Tu dataset registra datos cada 6 horas:
# Si 't' es la madrugada que queremos predecir, 'lag_1' es 6 horas antes (la noche previa)
# y 'lag_2' es 12 horas antes (la tarde previa).
for var in variables_clave:
    df[f'{var}_lag_1'] = df[var].shift(1)  # t - 6 horas
    df[f'{var}_lag_2'] = df[var].shift(2)  # t - 12 horas

# Eliminamos las filas iniciales que quedan vacías debido al desplazamiento temporal
df = df.dropna().reset_index(drop=True)

# Definimos X (Características del pasado) e y (Objetivo del presente/futuro)
features = [f'{var}_lag_1' for var in variables_clave] + [f'{var}_lag_2' for var in variables_clave]
X = df[features]
y = df['es_helada']


# =====================================================================
# 3. PASO 3: DIVISIÓN DE DATOS Y ESCALADO (OBLIGATORIO PARA SVM)
# =====================================================================
# Separamos el 80% para entrenar la IA y el 20% para evaluar qué tan buena es
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Como la presión atmosférica maneja números gigantes (Pascales) y la lluvia decimales,
# normalizamos los datos para que tengan la misma escala geométrica.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================================
# 4. PASO 4: ENTRENAMIENTO DEL MODELO SVM
# =====================================================================
print(f"-> Entrenando el clasificador SVM con {X_train_scaled.shape[0]} registros climáticos...")

# kernel='rbf': Permite trazar curvas de decisión climáticas complejas en el espacio multidimensional.
# class_weight='balanced': Obliga al modelo a prestar la misma atención a los días de helada, 
# evitando que se vuelva "perezoso" por el exceso de días normales de verano.
modelo_svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
modelo_svm.fit(X_train_scaled, y_train)


# =====================================================================
# 5. PASO 5: REPORTE DE RESULTADOS Y PRECISIÓN
# =====================================================================
print("\n" + "="*50)
print("      REPORTE DE RENDIMIENTO DEL MODELO SVM")
print("="*50)

# El modelo intenta predecir el 20% de datos que separamos para la prueba
y_pred = modelo_svm.predict(X_test_scaled)

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nMétricas Detalladas de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Helada (0)', 'Helada (1)']))