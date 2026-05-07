import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ENSEMBLE MODEL: XGBoost + Random Forest + LSTM")
print("="*60)

# Cargar datos
df = pd.read_csv('datos_heladas_altiplano.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['station', 'date']).reset_index(drop=True)

# Features
df['day_of_year'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Lags
for lag in [1, 2, 3]:
    df[f'T2M_MIN_lag_{lag}'] = df.groupby('station')['T2M_MIN'].shift(lag)

df = df.dropna()

feature_cols = [
    'elevation', 'day_of_year', 'month',
    'T2M_MAX', 'T2M_RANGE', 'RH2M', 'WS2M', 'PS', 'PRECTOTCORR',
    'T2M_MIN_lag_1', 'T2M_MIN_lag_2', 'T2M_MIN_lag_3'
]

X = df[feature_cols]
y = df['T2M_MIN']

# Division temporal
train_mask = df['year'] < 2024
test_mask = df['year'] >= 2024

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTrain: {len(X_train)} muestras")
print(f"Test: {len(X_test)} muestras")

# ==========================================
# 1. RANDOM FOREST
# ==========================================
print("\n" + "-"*40)
print("1. Entrenando Random Forest...")
print("-"*40)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest RMSE: {rmse_rf:.3f}°C")

# ==========================================
# 2. XGBOOST
# ==========================================
print("\n" + "-"*40)
print("2. Entrenando XGBoost...")
print("-"*40)

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb:.3f}°C")

# ==========================================
# 3. LSTM (PyTorch con GPU si disponible)
# ==========================================
print("\n" + "-"*40)
print("3. Entrenando LSTM...")
print("-"*40)

# Detectar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Preparar datos para LSTM (secuencias)
def create_sequences(X, y, seq_length=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# Normalizar
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Crear secuencias
seq_length = 7
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

# Dividir secuencias
train_size = int(len(X_seq) * 0.8)
X_seq_train, X_seq_test = X_seq[:train_size], X_seq[train_size:]
y_seq_train, y_seq_test = y_seq[:train_size], y_seq[train_size:]

# Convertir a tensores
X_seq_train = torch.FloatTensor(X_seq_train).to(device)
X_seq_test = torch.FloatTensor(X_seq_test).to(device)
y_seq_train = torch.FloatTensor(y_seq_train).to(device)
y_seq_test = torch.FloatTensor(y_seq_test).to(device)

# Definir modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_size = X_seq_train.shape[2]
lstm = LSTMModel(input_size).to(device)

# Entrenar LSTM
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

batch_size = 64
train_dataset = TensorDataset(X_seq_train, y_seq_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Entrenando LSTM (puede tomar unos minutos)...")
lstm.train()
for epoch in range(50):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = lstm(batch_X).flatten()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/50, Loss: {total_loss/len(train_loader):.4f}")

# Evaluar LSTM
lstm.eval()
with torch.no_grad():
    y_pred_lstm_scaled = lstm(X_seq_test).cpu().flatten().numpy()
    y_test_lstm_scaled = y_seq_test.cpu().numpy()

y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
y_test_lstm = scaler_y.inverse_transform(y_test_lstm_scaled.reshape(-1, 1)).flatten()

rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))
print(f"LSTM RMSE: {rmse_lstm:.3f}°C")

# ==========================================
# 4. ENSEMBLE (Weighted Average)
# ==========================================
print("\n" + "="*40)
print("ENSEMBLE - Weighted Average")
print("="*40)

# Alinear predicciones (LSTM tiene menos muestras por las secuencias)
min_len = min(len(y_pred_rf), len(y_pred_xgb), len(y_pred_lstm))
y_test_aligned = y_test[:min_len]
y_pred_rf_aligned = y_pred_rf[:min_len]
y_pred_xgb_aligned = y_pred_xgb[:min_len]
y_pred_lstm_aligned = y_pred_lstm[:min_len]

# Pesos optimos (se pueden ajustar)
weights = {'rf': 0.2, 'xgb': 0.5, 'lstm': 0.3}
y_pred_ensemble = (weights['rf'] * y_pred_rf_aligned + 
                   weights['xgb'] * y_pred_xgb_aligned + 
                   weights['lstm'] * y_pred_lstm_aligned)

rmse_ensemble = np.sqrt(mean_squared_error(y_test_aligned, y_pred_ensemble))
print(f"\nPesos: RF={weights['rf']}, XGB={weights['xgb']}, LSTM={weights['lstm']}")
print(f"Ensemble RMSE: {rmse_ensemble:.3f}°C")

# ==========================================
# 5. COMPARACION FINAL
# ==========================================
print("\n" + "="*40)
print("RESUMEN FINAL")
print("="*40)

results = pd.DataFrame({
    'Modelo': ['Random Forest', 'XGBoost', 'LSTM', 'Ensemble'],
    'RMSE (°C)': [rmse_rf, rmse_xgb, rmse_lstm, rmse_ensemble]
})
print(results.to_string(index=False))

print("\n" + "="*40)
print("COMPARACION CON EL PAPER")
print("="*40)
print(f"Paper - Random Forest: 1.83°C")
print(f"Paper - XGBoost: 1.78°C")
print(f"Paper - Ensemble: 1.65°C")
print(f"\nTu Ensemble RMSE: {rmse_ensemble:.3f}°C")

# ==========================================
# 6. GUARDAR PREDICCIONES DEL ENSEMBLE
# ==========================================
df_test = df[test_mask].iloc[:min_len].copy()
df_test['T2M_MIN_pred_ensemble'] = y_pred_ensemble
df_test[['station', 'elevation', 'date', 'T2M_MIN', 'T2M_MIN_pred_ensemble']].to_csv('predictions_ensemble.csv', index=False)

print("\n" + "="*40)
print("ARCHIVOS GUARDADOS:")
print("- predictions_ensemble.csv (predicciones del ensemble)")
print("="*40)