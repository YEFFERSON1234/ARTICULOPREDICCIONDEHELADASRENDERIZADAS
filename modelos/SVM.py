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
# 1. PASO 1: CONFIGURAR RUTA DINÁMICA Y CARGAR ARCHIVOS .NC
# =====================================================================
print("-> Iniciando el sistema de predicción de heladas...")
print("-> Calculando ruta dinámica relativa al script...")

# Detecta automáticamente dónde está 'svm.py' (dentro de /modelos)
ruta_actual_script = os.path.dirname(os.path.abspath(__file__))

# Sube un nivel (sale de 'modelos') y entra a 'datos_era5_puno'
carpeta_datos = os.path.abspath(os.path.join(ruta_actual_script, '..', 'datos_era5_puno'))
archivos = glob.glob(os.path.join(carpeta_datos, "*.nc"))

if not archivos:
    raise FileNotFoundError(
        f"\n[!] ERROR: No se encontraron archivos .nc en la ruta detectada:\n"
        f"    {carpeta_datos}\n"
        f"    Por favor, asegúrate de que la carpeta 'datos_era5_puno' tenga ese nombre exacto\n"
        f"    y esté ubicada al mismo nivel que la carpeta 'modelos'."
    )

print(f"[OK] Se encontraron {len(archivos)} archivos mensuales para procesar.")
print("-> Leyendo datos multidimensionales (unificando meses)...")

# 'join=override' y 'compat=override' ignoran diferencias decimales microscópicas de las coordenadas
ds = xr.open_mfdataset(archivos, combine='by_coords', coords='minimal', compat='override', join='override')

# Convertimos las temperaturas de Kelvin a Celsius de inmediato
ds['t2m'] = ds['t2m'] - 273.15
ds['d2m'] = ds['d2m'] - 273.15

print("-> Calculando promedio regional para la región de Puno...")
# Promediamos espacialmente la cuadrícula para consolidar las provincias en una línea de tiempo
ds_promedio = ds.mean(dim=['latitude', 'longitude'])

# Limpiamos remanentes geográficos para que Pandas no genere conflictos de índices duplicados
if 'latitude' in ds_promedio.coords:
    ds_promedio = ds_promedio.drop_vars('latitude')
if 'longitude' in ds_promedio.coords:
    ds_promedio = ds_promedio.drop_vars('longitude')

# Convertimos el cubo de datos a una tabla plana (DataFrame) de Pandas
df = ds_promedio.to_dataframe().reset_index()

# -----------------------------------------------------------------
# SOLUCIÓN AL KEYERROR: Detectar automáticamente la columna de tiempo
# -----------------------------------------------------------------
columnas_tiempo_probables = ['time', 'valid_time', 'Date', 'date']
columna_tiempo_real = None

for col in columnas_tiempo_probables:
    if col in df.columns:
        columna_tiempo_real = col
        break

if columna_tiempo_real is not None:
    # Renombramos la columna encontrada a 'time' para mantener la consistencia del script
    df = df.rename(columns={columna_tiempo_real: 'time'})
    print(f"[OK] Columna de tiempo detectada correctamente como: '{columna_tiempo_real}'")
else:
    raise KeyError(
        f"\n[!] ERROR: No se encontró ninguna columna de tiempo estándar en el archivo.\n"
        f"    Las columnas disponibles en tu archivo son: {list(df.columns)}\n"
        f"    Modifica la lista 'columnas_tiempo_probables' agregando el nombre correcto."
    )

# Ordenamos el DataFrame cronológicamente usando la columna normalizada
df = df.sort_values('time').reset_index(drop=True)


# =====================================================================
# 2. PASO 2: INGENIERÍA DE CARACTERÍSTICAS (PREPARACIÓN PARA PREVISIÓN)
# =====================================================================
print("-> Creando desfases temporales (Lags) para predecir el futuro...")

# Creamos la etiqueta binaria de helada: 1 si la temperatura es <= 0°C, 0 si no.
df['es_helada'] = (df['t2m'] <= 0).astype(int)

# Variables meteorológicas que el modelo analizará
variables_clave = ['t2m', 'd2m', 'sp', 'tp', 'ssrd']

# Tu dataset registra datos cada 6 horas:
# Si 't' es el momento que queremos predecir (madrugada), 'lag_1' son los datos de hace 6 horas (noche previa)
# y 'lag_2' representa los datos de hace 12 horas (tarde previa).
for var in variables_clave:
    if var in df.columns:
        df[f'{var}_lag_1'] = df[var].shift(1)  # t - 6 horas
        df[f'{var}_lag_2'] = df[var].shift(2)  # t - 12 horas
    else:
        raise KeyError(f"[!] La variable '{var}' no se encuentra en el dataset. Verifica sus nombres.")

# Eliminamos las filas iniciales que quedan vacías debido al desplazamiento temporal (shift)
df = df.dropna().reset_index(drop=True)

# Definimos X (Características del pasado) e y (Objetivo a predecir)
features = [f'{var}_lag_1' for var in variables_clave] + [f'{var}_lag_2' for var in variables_clave]
X = df[features]
y = df['es_helada']


# =====================================================================
# 3. PASO 3: DIVISIÓN DE DATOS Y ESCALADO (OBLIGATORIO PARA SVM)
# =====================================================================
# Separamos el 80% para entrenar la Inteligencia Artificial y el 20% para evaluar su rendimiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizamos las escalas para que la presión atmosférica y la lluvia no descalibren la SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================================
# 4. PASO 4: ENTRENAMIENTO DEL MODELO SVM
# =====================================================================
print(f"-> Entrenando el clasificador SVM con {X_train_scaled.shape[0]} registros climáticos...")

# kernel='rbf' y class_weight='balanced' para controlar el desbalance de heladas de Puno
modelo_svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
modelo_svm.fit(X_train_scaled, y_train)


# =====================================================================
# 5. PASO 5: REPORTE DE RESULTADOS Y METRICAS DE PRECISIÓN
# =====================================================================
print("\n" + "="*50)
print("      REPORTE DE RENDIMIENTO DEL MODELO SVM")
print("="*50)

# El modelo intenta predecir el 20% de datos de prueba retenidos
y_pred = modelo_svm.predict(X_test_scaled)

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nMétricas Detalladas de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Helada (0)', 'Helada (1)']))