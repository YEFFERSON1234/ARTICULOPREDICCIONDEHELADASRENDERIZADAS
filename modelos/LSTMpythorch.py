import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

# 1. CARGA Y PREPARACIÓN (Misma lógica anterior)
df = pd.read_csv('limpiezadedatos/datos_heladas_puno_REAL.csv').dropna(subset=['lat', 'lon', 'precip', 'tmax', 'tmin'])
df = df.sort_values(['estacion', 'fecha'])

features = ['lat', 'lon', 'precip', 'tmax', 'tmin']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length, -1])
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 7
X, y = create_sequences(df_scaled, SEQ_LENGTH)
split = int(len(X) * 0.8)
X_train, X_test = torch.FloatTensor(X[:split]), torch.FloatTensor(X[split:])
y_train, y_test = torch.FloatTensor(y[:split]), torch.FloatTensor(y[split:])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=False)

# 2. MODELO Y ENTRENAMIENTO (20 ÉPOCAS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FrostLSTM(input_size=5, hidden_size=64, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

print(f"Entrenando LSTM por 20 épocas...")
for epoch in range(20):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    print(f'Época [{epoch+1}/20], Loss: {loss.item():.6f}')

# 3. GUARDAR RESULTADOS
torch.save(model.state_dict(), 'modelos/lstm_puno_v1.pth')

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test.to(device)).cpu().numpy()

# Desnormalizar
dummy = np.zeros((len(y_pred_scaled), 5))
dummy[:, -1] = y_pred_scaled.flatten()
y_pred_final = scaler.inverse_transform(dummy)[:, -1]

# Guardar CSV para OpenGL y Ensamble
res = df.iloc[split+SEQ_LENGTH:].copy()
res['tmin_pred_lstm'] = y_pred_final
res.to_csv('limpiezadedatos/predictions_lstm.csv', index=False)
print("Archivo 'predictions_lstm.csv' guardado.")