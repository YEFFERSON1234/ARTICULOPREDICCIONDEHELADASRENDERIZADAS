import pandas as pd
import numpy as np
import sys

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Cargando datos SENAMHI...")
df = pd.read_csv('data_process/datos_heladas_puno_REAL.csv')

# Convertir fecha
df['fecha'] = pd.to_datetime(df['fecha'])

# Calcular amplitud térmica (tmax - tmin)
df['amp_termica'] = df['tmax'] - df['tmin']

# Calcular helada (1 si tmin <= 0, 0 si no)
df['helada'] = (df['tmin'] <= 0).astype(int)

# Guardar datos actualizados
df.to_csv('data_process/datos_heladas_puno_REAL.csv', index=False)

print("[OK] Datos actualizados con columnas amp_termica y helada")
print(f"Total de registros: {len(df)}")
print(f"Columnas: {list(df.columns)}")
print(f"\nEstadísticas de heladas:")
print(df['helada'].value_counts())
print(f"\nPrimeras filas:")
print(df.head())
