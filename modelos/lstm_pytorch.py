"""
lstm_pytorch.py
Modelo LSTM para predicción de heladas en Puno
Corregido: Segmentación por estaciones y alineación exacta de índices
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. ARQUITECTURA DEL MODELO
# ==========================================
class FrostLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FrostLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 2. PREPARACIÓN DE DATOS REESTRUCTURADA
# ==========================================
print("======================================================================")
#
print("MODELO LSTM PYTORCH - PUNO")
print("======================================================================")

cols_senamhi = ['lat', 'lon', 'precip', 'tmax', 'tmin']
df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv').dropna(subset=cols_senamhi)

df['fecha'] = pd.to_datetime(df['fecha'])
df['year'] = df['fecha'].dt.year
df = df.sort_values(['estacion', 'fecha']).reset_index(drop=True)

scaler = StandardScaler()
df_scaled_features = scaler.fit_transform(df[cols_senamhi])

df_scaled_df = pd.DataFrame(df_scaled_features, columns=cols_senamhi)
df_scaled_df['estacion'] = df['estacion'].values
df_scaled_df['year'] = df['year'].values
df_scaled_df['indice_original'] = df.index.values

# CREACIÓN DE SECUENCIAS EVITANDO MEZCLA DE PROVINCIAS
SEQ_LENGTH = 7
X_list, y_list, idx_list, year_list = [], [], [], []

for estacion, group in df_scaled_df.groupby('estacion'):
    group_values = group[cols_senamhi].values
    indices_originales = group['indice_original'].values
    years_originales = group['year'].values
    
    if len(group_values) <= SEQ_LENGTH:
        continue
        
    for i in range(len(group_values) - SEQ_LENGTH):
        X_list.append(group_values[i:(i + SEQ_LENGTH)])
        y_list.append(group_values[i + SEQ_LENGTH, -1])  # Target: tmin
        idx_list.append(indices_originales[i + SEQ_LENGTH])
        year_list.append(years_originales[i + SEQ_LENGTH])

X_all = np.array(X_list)
y_all = np.array(y_list)
idx_all = np.array(idx_list)
year_all = np.array(year_list)

# División Temporal Sincronizada (Prueba >= 2015)
mask_train = year_all < 2015
mask_test = year_all >= 2015

X_train = torch.FloatTensor(X_all[mask_train])
y_train = torch.FloatTensor(y_all[mask_train])
X_test = torch.FloatTensor(X_all[mask_test])
y_test = torch.FloatTensor(y_all[mask_test])
idx_test_real = idx_all[mask_test]  # Guardamos los punteros exactos del DataFrame original

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=False)

# ==========================================
# 3. CONFIGURACIÓN Y ENTRENAMIENTO
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FrostLSTM(input_size=5, hidden_size=64, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

print(f"\nEntrenando en {device} por 20 épocas...")
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f'Época [{epoch+1}/20], Loss: {total_loss/len(train_loader):.6f}')

# ==========================================
# 4. EVALUACIÓN Y MAPEO PRECISO (SOLUCIÓN AL ERROR)
# ==========================================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test.to(device)).cpu().numpy()

# Desnormalizar tmin predicha
dummy = np.zeros((len(y_pred_scaled), 5))
dummy[:, -1] = y_pred_scaled.flatten()
y_pred_final = scaler.inverse_transform(dummy)[:, -1]

# SOLUCIÓN DEFINITIVA AL VALUERROR: 
# Extraemos exactamente las filas correspondientes usando el array de índices guardado
res = df.loc[idx_test_real].copy()
res['prob_helada_lstm'] = y_pred_final  # Nombre estandarizado para tu script de ensamble

# Guardar archivos listos para el ensamble integrado
os.makedirs('data_process', exist_ok=True)
os.makedirs('modelos', exist_ok=True)
res.to_csv('data_process/predictions_lstm.csv', index=False)
torch.save(model.state_dict(), 'modelos/lstm_puno_v1.pth')

print("\n" + "="*70)
print(f"[OK] ¡Procesamiento completado con éxito!")
print(f"Registros guardados en test: {len(res)}")
print(f"RMSE LSTM (Escala Normalizada): {np.sqrt(mean_squared_error(y_test.numpy(), y_pred_scaled)):.4f}")
print(f"Archivo exportado: data_process/predictions_lstm.csv")
print("="*70)