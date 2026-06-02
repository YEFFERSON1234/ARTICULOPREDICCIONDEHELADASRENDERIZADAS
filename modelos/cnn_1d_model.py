"""
cnn1d_model.py
Modelo CNN-1D (Convolutional Neural Network 1D) para predicción de heladas
Red neuronal convolucional para series temporales con PyTorch
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import sys

# Configurar encoding para Windows
if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_cnn1d():
    """Entrena modelo CNN-1D para predicción de heladas"""
    print("="*70)
    print("MODELO CNN-1D")
    print("="*70)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("[ERROR] PyTorch no está instalado")
        print("Instala con: pip install torch")
        return
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos SENAMHI...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 2. Ingeniería de características
    print("[2/5] Creando características...")
    df['day_of_year'] = df['fecha'].dt.dayofyear
    df['month'] = df['fecha'].dt.month
    df['year'] = df['fecha'].dt.year
    
    # Lags temporales
    for lag in [1, 2, 3, 4, 5, 6, 7]:
        df[f'tmin_lag_{lag}'] = df.groupby('estacion')['tmin'].shift(lag)
    
    df = df.dropna()
    
    # Variables predictoras
    feature_cols = ['lat', 'lon', 'day_of_year', 'month', 'precip', 'tmax', 
                    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3', 'tmin_lag_4', 
                    'tmin_lag_5', 'tmin_lag_6', 'tmin_lag_7', 'amp_termica']
    
    X = df[feature_cols].values
    y = df['helada'].values
    
    # 3. Escalado
    print("[3/5] Escalando características...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Crear secuencias para CNN-1D
    SEQ_LENGTH = 7  # Ventana de 7 días
    X_seq = []
    y_seq = []
    
    for i in range(len(X_scaled) - SEQ_LENGTH):
        X_seq.append(X_scaled[i:i+SEQ_LENGTH])
        y_seq.append(y[i+SEQ_LENGTH])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Dividir temporalmente
    train_size = int(len(X_seq) * 0.8)
    X_train = torch.FloatTensor(X_seq[:train_size])
    X_test = torch.FloatTensor(X_seq[train_size:])
    y_train = torch.FloatTensor(y_seq[:train_size])
    y_test = torch.FloatTensor(y_seq[train_size:])
    
    print(f"Entrenamiento: {len(X_train)} secuencias")
    print(f"Prueba: {len(X_test)} secuencias")
    
    # 5. Definir modelo CNN-1D
    print("[4/5] Definiendo CNN-1D...")
    
    class CNN1D(nn.Module):
        def __init__(self, input_size):
            super(CNN1D, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            # x: (batch, seq_len, features) -> (batch, features, seq_len)
            x = x.transpose(1, 2)
            
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            
            x = x.view(x.size(0), -1)  # Flatten
            x = self.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x
    
    model = CNN1D(input_size=X_train.shape[2])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Entrenamiento
    print("[5/5] Entrenando CNN-1D...")
    num_epochs = 50
    batch_size = 32
    
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # 7. Evaluación
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test).squeeze().numpy()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        y_test_np = y_test.numpy()
    
    f1 = f1_score(y_test_np, y_pred)
    auc = roc_auc_score(y_test_np, y_pred_prob)
    
    print(f"\n" + "="*70)
    print(f"RESULTADOS CNN-1D")
    print(f"="*70)
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Guardar modelo
    torch.save(model.state_dict(), 'modelos/cnn1d_puno_v1.pth')
    
    # Guardar predicciones
    # Necesitamos alinear con el DataFrame original
    df_test = df.iloc[train_size + SEQ_LENGTH:].copy()
    df_test['prob_helada_cnn1d'] = y_pred_prob[:len(df_test)]
    df_test.to_csv('data_process/predictions_cnn1d.csv', index=False)
    
    print(f"\n[OK] Predicciones CNN-1D guardadas en: data_process/predictions_cnn1d.csv")
    print(f"Modelo guardado en: modelos/cnn1d_puno_v1.pth")

if __name__ == '__main__':
    train_cnn1d()
