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
        # Capa LSTM para capturar dependencias temporales
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Capa Lineal para la salida de temperatura (°C)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # out: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)
        # Solo usamos el último paso de tiempo para predecir el día siguiente
        return self.fc(out[:, -1, :])

# ==========================================
# 2. PREPARACIÓN DE DATOS (SENAMHI PUNO)
# ==========================================
print("Cargando base de datos real de Puno...")
# Columnas basadas en el formato estándar de SENAMHI [cite: 403-408]
cols_senamhi = ['lat', 'lon', 'precip', 'tmax', 'tmin']
df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv').dropna(subset=cols_senamhi)

# Asegurar orden cronológico por estación
df['fecha'] = pd.to_datetime(df['fecha'])
df['year'] = df['fecha'].dt.year
df = df.sort_values(['estacion', 'fecha'])

# Normalización: Crucial para evitar "Loss: nan" en PyTorch
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[cols_senamhi])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length, -1]) # Target: tmin
    return np.array(xs), np.array(ys)

# Ventana de 7 días (Memoria semanal)
SEQ_LENGTH = 7
X_all, y_all = create_sequences(df_scaled, SEQ_LENGTH)

# División Temporal: Sincronización con XGBoost/RF (Prueba >= 2015)
mask_test = df['year'].values[SEQ_LENGTH:] >= 2015
mask_train = df['year'].values[SEQ_LENGTH:] < 2015

X_train = torch.FloatTensor(X_all[mask_train])
y_train = torch.FloatTensor(y_all[mask_train])
X_test = torch.FloatTensor(X_all[mask_test])
y_test = torch.FloatTensor(y_all[mask_test])

# DataLoader para optimizar VRAM de 2GB
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=False)

# ==========================================
# 3. CONFIGURACIÓN Y ENTRENAMIENTO
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FrostLSTM(input_size=5, hidden_size=64, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

print(f"Entrenando en {device} por 20 épocas...")
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        
        loss.backward()
        # Gradient Clipping: Evita que los números exploten
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f'Época [{epoch+1}/20], Loss: {total_loss/len(train_loader):.6f}')

# ==========================================
# 4. EVALUACIÓN Y GUARDADO SINCRONIZADO
# ==========================================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test.to(device)).cpu().numpy()

# Desnormalizar resultados
dummy = np.zeros((len(y_pred_scaled), 5))
dummy[:, -1] = y_pred_scaled.flatten()
y_pred_final = scaler.inverse_transform(dummy)[:, -1]

# ALINEACIÓN DE ÍNDICE: Evita el error de "Length Mismatch"
df_test_final = df[df['year'] >= 2015].copy()
# La primera predicción ocurre después de la primera ventana de SEQ_LENGTH
res = df_test_final.iloc[SEQ_LENGTH:].copy()

# Recorte de seguridad para asegurar coincidencia exacta
res = res.head(len(y_pred_final))
res['tmin_pred_lstm'] = y_pred_final

# Guardar para el Ensamble y OpenGL
os.makedirs('data_process', exist_ok=True)
res.to_csv('data_process/predictions_lstm.csv', index=False)
torch.save(model.state_dict(), 'modelos/lstm_puno_v1.pth')

print(f"\n¡Éxito! Registros generados: {len(res)}")
print(f"RMSE LSTM: {np.sqrt(mean_squared_error(y_test.numpy(), y_pred_scaled)):.4f} (Normalizado)")