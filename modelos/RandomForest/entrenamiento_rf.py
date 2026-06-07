import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils.data_loading import load_senamhi_csvs, get_project_paths
from utils.evaluation import compute_classification_metrics, print_table_metrics
from utils.plotting import plot_confusion_matrix, plot_roc_curve

# 1. CONFIGURACION DE RUTAS
paths = get_project_paths(__file__)

# 2. CARGA Y PREPARACION DE DATOS (SENAMHI)
print("-> [1/5] Cargando datos historicos de SENAMHI...")
df_senamhi = load_senamhi_csvs(paths['senamhi_dir'])

X = df_senamhi[['mes', 'precipitacion']]
y = df_senamhi['helada']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. ENTRENAMIENTO DEL MODELO (RANDOM FOREST)
print("-> [2/5] Entrenando Clasificador Random Forest...")
model_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)
y_prob = model_rf.predict_proba(X_test)[:, 1]

# 4. METRICAS PARA TABLA II
metrics = compute_classification_metrics(y_test, y_pred, y_prob)
print_table_metrics("Random Forest (Frost Detection)", metrics)

# 5. GRAFICOS ANALITICOS
print("-> [3/5] Generando y guardando curvas analiticas...")
plot_confusion_matrix(
    y_test, y_pred, 'Random Forest',
    os.path.join(paths['plots_dir'], 'matriz_confusion.png'),
    cmap='Blues',
)
plot_roc_curve(
    y_test, y_prob, 'Random Forest',
    os.path.join(paths['plots_dir'], 'curva_roc.png'),
    color='darkorange',
)

# 6. PREDICCION SOBRE LA GRILLA DE ERA5 Y EXPORTACION FINAL
print("-> [4/5] Aplicando el modelo sobre la grilla climatica de ERA5...")
archivos_era5 = glob.glob(os.path.join(paths['era5_dir'], "*.csv"))

if not archivos_era5:
    print(f"[!] Alerta: No se encontraron archivos de ERA5 en {paths['era5_dir']}.")
    exit()

lista_predicciones = []

for f in archivos_era5:
    df_era5 = pd.read_csv(f)
    df_era5['valid_time'] = pd.to_datetime(df_era5['valid_time'])
    df_era5['fecha'] = df_era5['valid_time'].dt.date
    df_era5['mes'] = df_era5['valid_time'].dt.month
    df_era5['t2m_celsius'] = df_era5['t2m'] - 273.15
    df_era5['tp_mm'] = df_era5['tp'] * 1000

    df_diario = df_era5.groupby(['latitude', 'longitude', 'fecha', 'mes']).agg(
        temp_min=('t2m_celsius', 'min'),
        precipitacion=('tp_mm', 'sum')
    ).reset_index()

    X_era5 = df_diario[['mes', 'precipitacion']]
    df_diario['prob_helada'] = model_rf.predict_proba(X_era5)[:, 1]

    df_final_mes = df_diario[['latitude', 'longitude', 'fecha', 'prob_helada', 'temp_min']].copy()
    df_final_mes = df_final_mes.rename(columns={'latitude': 'Lat', 'longitude': 'Long'})
    lista_predicciones.append(df_final_mes)

print("-> [5/5] Exportando resultados finales a la carpeta Predicciones...")
if lista_predicciones:
    df_predictions_total = pd.concat(lista_predicciones, ignore_index=True)
    ruta_salida_predictions = os.path.join(paths['predictions_dir'], 'predictions.csv')
    df_predictions_total.to_csv(ruta_salida_predictions, index=False)

    plt.figure(figsize=(6, 5))
    plt.scatter(df_predictions_total['temp_min'], df_predictions_total['prob_helada'], alpha=0.02, color='teal')
    plt.axvline(x=0, color='red', linestyle='--', label='Umbral de Helada (0 C)')
    plt.xlabel('Temperatura Minima Diaria ERA5 (C)')
    plt.ylabel('Probabilidad de Helada')
    plt.title('Grafica de Dispersion: Probabilidad vs Temp Minima (ERA5)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(paths['plots_dir'], 'grafico_dispersion.png'))
    plt.close()
    print(f"Exito total! Todo el pipeline se ejecuto correctamente.")
else:
    print("[!] Error critico: No se procesaron archivos de prediccion.")
