import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

# =====================================================================
# 1. CONFIGURACIÓN DE RUTAS RELATIVAS (ESTRUCTURA DEL PROYECTO)
# =====================================================================
dir_rf = os.path.dirname(os.path.abspath(__file__))
raiz_proyecto = os.path.abspath(os.path.join(dir_rf, "..", ".."))

carpeta_senamhi = os.path.join(raiz_proyecto, 'data', 'datos_senamhi_puno_csv')
carpeta_era5 = os.path.join(raiz_proyecto, 'data', 'datos_era5_puno_csv')
carpeta_predicciones = os.path.join(raiz_proyecto, 'Predicciones')
carpeta_graficos = os.path.join(dir_rf, 'graficos_resultados')

os.makedirs(carpeta_predicciones, exist_ok=True)
os.makedirs(carpeta_graficos, exist_ok=True)

# =====================================================================
# 2. CARGA Y PREPARACIÓN DE DATOS (SENAMHI)
# =====================================================================
print("-> [1/5] Cargando datos históricos de SENAMHI...")
archivos_senamhi = glob.glob(os.path.join(carpeta_senamhi, "*.csv"))

if not archivos_senamhi:
    print(f"[!] Error: No se encontraron archivos .csv en {carpeta_senamhi}")
    exit()

list_df_senamhi = []
for f in archivos_senamhi:
    df_temp = pd.read_csv(f)
    df_temp['estacion'] = os.path.basename(f).replace('.csv', '')
    list_df_senamhi.append(df_temp)

df_senamhi = pd.concat(list_df_senamhi, ignore_index=True)

# Limpieza: Aseguramos que haya datos en 'temp_min' y 'precipitacion'
df_senamhi = df_senamhi.dropna(subset=['temp_min', 'precipitacion'])

# Definición del Target Binario (Helada: 1 si es menor o igual a 0°C)
df_senamhi['helada'] = (df_senamhi['temp_min'] <= 0).astype(int)

# Variables predictoras para evitar Overfitting colineal
X = df_senamhi[['mes', 'precipitacion']] 
y = df_senamhi['helada']

# Partición balanceada 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =====================================================================
# 3. ENTRENAMIENTO DEL MODELO (RANDOM FOREST)
# =====================================================================
print("-> [2/5] Entrenando Clasificador Random Forest...")
model_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)
y_prob = model_rf.predict_proba(X_test)[:, 1]

# =====================================================================
# 4. GENERACIÓN DE MÉTRICAS PARA TU TABLA II
# =====================================================================
print("\n" + "="*45 + "\n=== EXTRAE ESTOS DATOS PARA TU TABLA II ===\n" + "="*45)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
tss = (tp / (tp + fn)) + (tn / (tn + fp)) - 1

print(f"Modelo: Random Forest (Frost Detection)")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")
print(f"TSS:       {tss:.3f}")
print("="*45 + "\n")

# =====================================================================
# 5. GENERACIÓN DE GRÁFICOS ANALÍTICOS
# =====================================================================
print("-> [3/5] Generando y guardando curvas analíticas...")

# Gráfico A: Matriz de Confusión
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Helada', 'Helada'], yticklabels=['No Helada', 'Helada'])
plt.title('Matriz de Confusión - Random Forest')
plt.ylabel('Realidad (SENAMHI)')
plt.xlabel('Predicción (Modelo)')
plt.tight_layout()
plt.savefig(os.path.join(carpeta_graficos, 'matriz_confusion.png'))
plt.close()

# Gráfico B: Curva ROC
fpr, tpr_roc, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr_roc)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr_roc, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Detección de Heladas en Puno')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_graficos, 'curva_roc.png'))
plt.close()

# =====================================================================
# 6. PREDICCIÓN SOBRE LA GRILLA DE ERA5 Y EXPORTACIÓN FINAL
# =====================================================================
print("-> [4/5] Aplicando el modelo sobre la grilla climática de ERA5...")
archivos_era5 = glob.glob(os.path.join(carpeta_era5, "*.csv"))

if not archivos_era5:
    print(f"[!] Alerta: No se encontraron archivos de ERA5 en {carpeta_era5}.")
    exit()

lista_predicciones = []

for f in archivos_era5:
    df_era5 = pd.read_csv(f)
    
    # CONTROL DE SEGURIDAD: Convertir columna de tiempo específica de tu archivo
    df_era5['valid_time'] = pd.to_datetime(df_era5['valid_time'])
    df_era5['fecha'] = df_era5['valid_time'].dt.date
    df_era5['mes'] = df_era5['valid_time'].dt.month
    
    # CONVERSIÓN DE UNIDADES CRUDAS DE ERA5 SOBRE LA MARCHA:
    # Kelvin a Celsius para la temperatura de almacenamiento
    df_era5['t2m_celsius'] = df_era5['t2m'] - 273.15
    # Metros a Milímetros para la precipitación
    df_era5['tp_mm'] = df_era5['tp'] * 1000
    
    # Agrupación por píxel para pasar de escala horaria a resumen diario
    df_diario = df_era5.groupby(['latitude', 'longitude', 'fecha', 'mes']).agg(
        temp_min=('t2m_celsius', 'min'),
        precipitacion=('tp_mm', 'sum')
    ).reset_index()
    
    # Alinear las columnas predictoras exactas
    X_era5 = df_diario[['mes', 'precipitacion']]
    
    # Calcular inferencias probabilísticas
    df_diario['prob_helada'] = model_rf.predict_proba(X_era5)[:, 1]
    
    # Renombrar columnas finales requeridas para tu mapa/archivo final
    df_final_mes = df_diario[['latitude', 'longitude', 'fecha', 'prob_helada', 'temp_min']].copy()
    df_final_mes = df_final_mes.rename(columns={'latitude': 'Lat', 'longitude': 'Long'})
    
    lista_predicciones.append(df_final_mes)

# Consolidar y exportar predictions.csv
print("-> [5/5] Exportando resultados finales a la carpeta Predicciones...")
if lista_predicciones:
    df_predictions_total = pd.concat(lista_predicciones, ignore_index=True)
    ruta_salida_predictions = os.path.join(carpeta_predicciones, 'predictions.csv')
    df_predictions_total.to_csv(ruta_salida_predictions, index=False)

    # Gráfico C: Dispersión Final
    plt.figure(figsize=(6, 5))
    plt.scatter(df_predictions_total['temp_min'], df_predictions_total['prob_helada'], alpha=0.02, color='teal')
    plt.axvline(x=0, color='red', linestyle='--', label='Umbral de Helada (0°C)')
    plt.xlabel('Temperatura Mínima Diaria ERA5 (°C)')
    plt.ylabel('Probabilidad de Helada')
    plt.title('Gráfica de Dispersión: Probabilidad vs Temp Mínima (ERA5)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_graficos, 'grafico_dispersion.png'))
    plt.close()
    print(f"¡Éxito total! Todo el pipeline se ejecutó correctamente sin errores de índice.")
else:
    print("[!] Error crítico: No se procesaron archivos de predicción.")