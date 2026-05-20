import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# =====================================================================
# 1. PASO 1: CONFIGURAR RUTA DINÁMICA Y CARGAR ARCHIVOS .NC
# =====================================================================
print("-> Iniciando el sistema de predicción de heladas...")
print("-> Calculando ruta dinámica relativa al script...")

ruta_actual_script = os.path.dirname(os.path.abspath(__file__))
carpeta_datos = os.path.abspath(os.path.join(ruta_actual_script, '..', 'datos_era5_puno'))
archivos = glob.glob(os.path.join(carpeta_datos, "*.nc"))

if not archivos:
    raise FileNotFoundError(
        f"\n[!] ERROR: No se encontraron archivos .nc en la ruta detectada:\n"
        f"    {carpeta_datos}\n"
        f"    Por favor, asegúrate de que la carpeta 'datos_era5_puno' tenga ese nombre exacto."
    )

print(f"[OK] Se encontraron {len(archivos)} archivos mensuales para procesar.")
print("-> Leyendo datos multidimensionales (unificando meses)...")

ds = xr.open_mfdataset(archivos, combine='by_coords', coords='minimal', compat='override', join='override')

ds['t2m'] = ds['t2m'] - 273.15
ds['d2m'] = ds['d2m'] - 273.15

print("-> Calculando promedio regional para la región de Puno...")
ds_promedio = ds.mean(dim=['latitude', 'longitude'])

if 'latitude' in ds_promedio.coords: ds_promedio = ds_promedio.drop_vars('latitude')
if 'longitude' in ds_promedio.coords: ds_promedio = ds_promedio.drop_vars('longitude')

df = ds_promedio.to_dataframe().reset_index()

# Auto-detección de la columna de tiempo
columnas_tiempo_probables = ['time', 'valid_time', 'Date', 'date']
columna_tiempo_real = None
for col in columnas_tiempo_probables:
    if col in df.columns:
        columna_tiempo_real = col
        break

if columna_tiempo_real is not None:
    df = df.rename(columns={columna_tiempo_real: 'time'})
    print(f"[OK] Columna de tiempo detectada correctamente como: '{columna_tiempo_real}'")
else:
    raise KeyError(f"\n[!] ERROR: No se encontró columna de tiempo. Columnas: {list(df.columns)}")

df = df.sort_values('time').reset_index(drop=True)


# =====================================================================
# 2. PASO 2: INGENIERÍA DE CARACTERÍSTICAS
# =====================================================================
print("-> Creando desfases temporales (Lags) para predecir el futuro...")

df['es_helada'] = (df['t2m'] <= 0).astype(int)
variables_clave = ['t2m', 'd2m', 'sp', 'tp', 'ssrd']

for var in variables_clave:
    if var in df.columns:
        df[f'{var}_lag_1'] = df[var].shift(1)  
        df[f'{var}_lag_2'] = df[var].shift(2)  
    else:
        raise KeyError(f"[!] La variable '{var}' no se encuentra en el dataset.")

df = df.dropna().reset_index(drop=True)

features = [f'{var}_lag_1' for var in variables_clave] + [f'{var}_lag_2' for var in variables_clave]
X = df[features]
y = df['es_helada']


# =====================================================================
# 3. PASO 3: DIVISIÓN DE DATOS Y ESCALADO
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================================
# 4. PASO 4: ENTRENAMIENTO DEL MODELO SVM
# =====================================================================
print(f"-> Entrenando el clasificador SVM con {X_train_scaled.shape[0]} registros climáticos...")
modelo_svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
modelo_svm.fit(X_train_scaled, y_train)


# =====================================================================
# 5. PASO 5: REPORTE DE RESULTADOS Y METRICAS
# =====================================================================
print("\n" + "="*50)
print("      REPORTE DE RENDIMIENTO DEL MODELO SVM")
print("="*50)

y_pred = modelo_svm.predict(X_test_scaled)
y_prob = modelo_svm.predict_proba(X_test_scaled)[:, 1]

print("\nMatriz de Confusión:")
matriz = confusion_matrix(y_test, y_pred)
print(matriz)

print("\nMétricas Detalladas de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Helada (0)', 'Helada (1)']))


# =====================================================================
# 6. PASO 6: GENERACIÓN Y GUARDADO DE GRÁFICOS PARA EL ARTÍCULO
# =====================================================================
print("\n-> Generando gráficos estadísticos y curvas de rendimiento...")

# Crear carpeta para almacenar las imágenes si no existe
carpeta_graficos = os.path.join(os.path.dirname(ruta_actual_script), 'graficos_resultados')
os.makedirs(carpeta_graficos, exist_ok=True)

# Configuración estética general para papers científicos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# --- GRÁFICO 1: Matriz de Confusión Visual ---
plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Helada (0)', 'Helada (1)'],
            yticklabels=['No Helada (0)', 'Helada (1)'])
plt.title('Matriz de Confusión - Predicción de Heladas (SVM)')
plt.ylabel('Clase Real (Observado)')
plt.xlabel('Clase Predicha (Modelo)')
plt.tight_layout()
ruta_grafico1 = os.path.join(carpeta_graficos, 'matriz_confusion.png')
plt.savefig(ruta_grafico1, dpi=300)
plt.close()
print(f"[OK] Matriz de confusión guardada en: {ruta_grafico1}")

# --- GRÁFICO 2: Curva ROC (Rendimiento del Clasificador) ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (Área = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.title('Curva ROC del Modelo SVM')
plt.legend(loc="lower right")
plt.tight_layout()
ruta_grafico2 = os.path.join(carpeta_graficos, 'curva_roc.png')
plt.savefig(ruta_grafico2, dpi=300)
plt.close()
print(f"[OK] Curva ROC guardada en: {ruta_grafico2}")

# --- GRÁFICO 3: Importancia Relativa de las Variables (Pesos de correlación) ---
# Al usar kernel RBF no hay coeficientes lineales directos, por lo que evaluamos la correlación 
# de las variables lag con el evento de helada para mostrar la importancia física en el paper.
correlaciones = df[features].corrwith(df['es_helada']).abs().sort_values(ascending=True)

plt.figure(figsize=(8, 5))
correlaciones.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('Importancia Predictiva Física (Correlación Absoluta con el Evento Helada)')
plt.xlabel('Coeficiente de Correlación Absoluto')
plt.ylabel('Variables Meteorológicas (Lags)')
plt.tight_layout()
ruta_grafico3 = os.path.join(carpeta_graficos, 'importancia_variables.png')
plt.savefig(ruta_grafico3, dpi=300)
plt.close()
print(f"[OK] Gráfico de variables guardado en: {ruta_grafico3}")

print(f"\n[PROCESO TERMINADO] Todos los gráficos de alta resolución (300 DPI) listos para tu artículo en: \n -> {carpeta_graficos}")