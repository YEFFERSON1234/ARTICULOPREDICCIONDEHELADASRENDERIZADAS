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
# 1. PASO 1: CONFIGURAR RUTA (UN NIVEL ARRIBA) Y CARGAR ARCHIVOS .NC
# =====================================================================
print("-> Configurando rutas y buscando archivos NetCDF...")

carpeta_datos = os.path.join('..', 'datos_era5_puno')
archivos = glob.glob(os.path.join(carpeta_datos, "*.nc"))

if not archivos:
    raise FileNotFoundError(
        f"\n[!] ERROR: No se encontraron archivos .nc en la ruta detectada:\n"
        f"    {os.path.abspath(carpeta_datos)}\n"
        f"    Por favor, verifica que el nombre de la carpeta sea el correcto."
    )

print(f"[OK] Se encontraron {len(archivos)} archivos mensuales para procesar.")
print("-> Leyendo datos multidimensionales...")

# SOLUCIÓN: Agregamos coords='minimal', compat='override' y join='override' 
# para evitar conflictos de dimensiones entre archivos mensuales.
ds = xr.open_mfdataset(archivos, combine='by_coords', coords='minimal', compat='override', join='override')

# Convertimos las temperaturas de Kelvin a Celsius
ds['t2m'] = ds['t2m'] - 273.15
ds['d2m'] = ds['d2m'] - 273.15

print("-> Calculando promedio regional...")
# Calculamos el promedio espacial para toda la región de Puno mapeada
ds_promedio = ds.mean(dim=['latitude', 'longitude'])

# SOLUCIÓN: Forzamos la limpieza de las dimensiones espaciales remanentes 
# para que Pandas no genere índices conflictivos o duplicados.
if 'latitude' in ds_promedio.coords:
    ds_promedio = ds_promedio.drop_vars('latitude')
if 'longitude' in ds_promedio.coords:
    ds_promedio = ds_promedio.drop_vars('longitude')

# Transformamos la estructura a un DataFrame indexado por fecha/hora (time)
df = ds_promedio.to_dataframe().reset_index()

# Ordenamos el DataFrame cronológicamente
df = df.sort_values('time').reset_index(drop=True)


# =====================================================================
# 2. PASO 2: INGENIERÍA DE CARACTERÍSTICAS (CREAR LAGS Y TARGET)
# =====================================================================
print("-> Construyendo variables predictoras (Pasado) y objetivo (Futuro)...")

df['es_helada'] = (df['t2m'] <= 0).astype(int)

variables_clave = ['t2m', 'd2m', 'sp', 'tp', 'ssrd']

for var in variables_clave:
    df[f'{var}_lag_1'] = df[var].shift(1)  # Hace 6 horas (t - 1)
    df[f'{var}_lag_2'] = df[var].shift(2)  # Hace 12 horas (t - 2)

df = df.dropna().reset_index(drop=True)

features = [f'{var}_lag_1' for var in variables_clave] + [f'{var}_lag_2' for var in variables_clave]
X = df[features]
y = df['es_helada']


# =====================================================================
# 3. PASO 3: DIVISIÓN DE DATOS Y NORMALIZACIÓN (ESCALADO)
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================================
# 4. PASO 4: CONFIGURACIÓN Y ENTRENAMIENTO DE LA SVM
# =====================================================================
print(f"-> Entrenando clasificador SVM con {X_train_scaled.shape[0]} registros...")

modelo_svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
modelo_svm.fit(X_train_scaled, y_train)


# =====================================================================
# 5. PASO 5: EVALUACIÓN DE RESULTADOS
# =====================================================================
print("\n" + "="*50)
print("      REPORTE DE RENDIMIENTO DEL MODELO SVM")
print("="*50)

y_pred = modelo_svm.predict(X_test_scaled)

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nMétricas Detalladas de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Helada (0)', 'Helada (1)']))