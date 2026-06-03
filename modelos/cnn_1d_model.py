"""
cnn1d_model.py
Modelo CNN-1D (Convolutional Neural Network 1D) para predicción de heladas
Red neuronal convolucional para series temporales con PyTorch - Versión Optimizada
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
    print("="*70)
    print("MODELO CNN-1D OPTIMIZADO - PUNO")
    print("="*70)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from tqdm import tqdm  # Asegúrate de tenerlo instalado: pip install tqdm
    except ImportError:
        print("[ERROR] PyTorch o tqdm no están instalados")
        print("Instala con: pip install torch tqdm")
        return
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos SENAMHI...")
    df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Asegurar el orden cronológico por cada estación
    df = df.sort_values(by=['estacion', 'fecha']).reset_index(drop=True)
    
    # 2. Ingeniería de características
    print("[2/5] Creando características...")
    df['day_of_year'] = df['fecha'].dt.dayofyear
    df['month'] = df['fecha'].dt.month
    df['year'] = df['fecha'].dt.year
    
    # Lags temporales correctos por estación
    for lag in [1, 2, 3, 4, 5, 6, 7]:
        df[f'tmin_lag_{lag}'] = df.groupby('estacion')['tmin'].shift(lag)
    
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = ['lat', 'lon', 'day_of_year', 'month', 'precip', 'tmax', 
                    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3', 'tmin_lag_4', 
                    'tmin_lag_5', 'tmin_lag_6', 'tmin_lag_7', 'amp_termica']
    
    # 3. Escalado
    print("[3/5] Escalando características...")
    scaler = StandardScaler()
    df_scaled_features = scaler.fit_transform(df[feature_cols])
    
    # Reincorporar al DataFrame para poder segmentar por estación con seguridad
    df_scaled = pd.DataFrame(df_scaled_features, columns=feature_cols)
    df_scaled['estacion'] = df['estacion'].values
    df_scaled['helada'] = df['helada'].values
    
    # 4. Crear secuencias para CNN-1D EVITANDO mezcla de estaciones
    print("[4/5] Generando secuencias temporales por estación...")
    SEQ_LENGTH = 7
    X_seq = []
    y_seq = []
    indices_originales = [] # Para mapear las predicciones al DataFrame original al final
    
    for estacion, group in df_scaled.groupby('estacion'):
        X_group = group[feature_cols].values
        y_group = group['helada'].values
        idx_group = group.index.values
        
        if len(X_group) <= SEQ_LENGTH:
            continue
            
        for i in range(len(X_group) - SEQ_LENGTH):
            X_seq.append(X_group[i:i+SEQ_LENGTH])
            y_seq.append(y_group[i+SEQ_LENGTH])
            indices_originales.append(idx_group[i+SEQ_LENGTH])
            
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    indices_originales = np.array(indices_originales)
    
    # Dividir temporalmente (80% tren, 20% test)
    train_size = int(len(X_seq) * 0.8)
    
    X_train = torch.FloatTensor(X_seq[:train_size])
    X_test = torch.FloatTensor(X_seq[train_size:])
    y_train = torch.FloatTensor(y_seq[:train_size])
    y_test = torch.FloatTensor(y_seq[train_size:])
    
    print(f"-> Entrenamiento: {len(X_train)} secuencias")
    print(f"-> Prueba: {len(X_test)} secuencias")
    
    # 5. Definir modelo CNN-1D
    class CNN1D(nn.Module):
        def __init__(self, input_size):
            super(CNN1D, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding='same')
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
            self.pool = nn.MaxPool1d(kernel_size=2)
            # Longitud 7 -> pool1 -> 3 -> pool2 -> 1
            self.fc1 = nn.Linear(32 * 1, 16)
            self.fc2 = nn.Linear(16, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = x.transpose(1, 2) # (batch, seq, feat) -> (batch, feat, seq)
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x
    
    model = CNN1D(input_size=X_train.shape[2])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Entrenamiento optimizado con barra de progreso
    print("\n[5/5] Entrenando CNN-1D...")
    num_epochs = 15  # 15 épocas es suficiente para converger con Adam en este volumen
    batch_size = 512 # Lotes más grandes aceleran masivamente el cómputo en CPU
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Barra de progreso visual por época
        with tqdm(range(0, len(X_train), batch_size), desc=f"Época {epoch+1}/{num_epochs}") as pbar:
            for i in pbar:
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(Loss=f"{loss.item():.4f}")
                
    # 7. Evaluación
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test).squeeze().numpy()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        y_test_np = y_test.numpy()
    
    f1 = f1_score(y_test_np, y_pred)
    auc = roc_auc_score(y_test_np, y_pred_prob)
    
    print("\n" + "="*70)
    print("RESULTADOS FINALES CNN-1D")
    print("="*70)
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Guardar modelo
    torch.save(model.state_dict(), 'modelos/cnn1d_puno_v1.pth')
    
    # Guardar predicciones mapeadas correctamente
    indices_test = indices_originales[train_size:]
    df_test = df.loc[indices_test].copy()
    df_test['prob_helada_cnn1d'] = y_pred_prob
    
    df_test.to_csv('data_process/predictions_cnn1d.csv', index=False)
    print(f"\n[OK] Modelo guardado en: modelos/cnn1d_puno_v1.pth")
    print(f"[OK] Predicciones reales guardadas en: data_process/predictions_cnn1d.csv")

if __name__ == '__main__':
    train_cnn1d()