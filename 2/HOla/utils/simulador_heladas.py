import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

def simular_heladas_desde_csv(ruta_csv, resolucion=100):
    print(f"   Cargando datos desde: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    print(f"   Columnas encontradas: {list(df.columns)}")

    if 'longitud' in df.columns and 'latitud' in df.columns:
        lon = df['longitud'].values
        lat = df['latitud'].values
        X, Y = np.meshgrid(
            np.linspace(lon.min(), lon.max(), resolucion),
            np.linspace(lat.min(), lat.max(), resolucion)
        )
        # Simular alturas realistas
        Z = np.sin(X * 10) * np.cos(Y * 10) * 600 + 3500
        Z += np.random.normal(0, 150, Z.shape)
    else:
        print("   ⚠️  Creando coordenadas genéricas...")
        X, Y = np.meshgrid(
            np.linspace(-71, -68, resolucion),
            np.linspace(-17, -14, resolucion)
        )
        Z = np.random.rand(resolucion, resolucion) * 2000 + 3000

    # Probabilidad de helada según altura
    Z_min, Z_max = Z.min(), Z.max()
    prob = (Z - Z_min) / (Z_max - Z_min)
    prob += np.random.normal(0, 0.05, Z.shape)
    prob = gaussian_filter(prob, sigma=1.5)
    prob = np.clip(prob, 0, 1)

    print(f"   ✅ Datos generados: {Z.shape}")
    return X, Y, Z, prob