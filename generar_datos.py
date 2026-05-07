import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

n_stations = 13
n_days = 365 * 25
start_date = datetime(2000, 1, 1)

stations = [
    "Puno", "Juliaca", "Azángaro", "Ayaviri", "Macusani", "Mazocruz",
    "Lampa", "Yunguyo", "Juli", "Desaguadero", "Cojata", "Crucero", "Crucero Alto"
]

elevations = [3825, 3824, 3859, 3918, 4345, 3990, 3872, 3826, 3812, 3808, 4320, 4130, 4470]

dates = [start_date + timedelta(days=i) for i in range(n_days)]

all_data = []

for station, elev in zip(stations, elevations):
    
    day_of_year = np.arange(n_days) % 365
    
    # Ciclo anual
    seasonal_temp = -6 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    
    # Tendencia
    warming_trend = 0.00005 * np.arange(n_days)
    
    # Altitud
    altitude_effect = -0.006 * (elev - 3800)
    
    # Ruido grande y no lineal
    noise = np.random.normal(0, 2.5, n_days)
    
    # Componente caotico (random walk)
    chaotic = np.cumsum(np.random.normal(0, 0.3, n_days)) * 0.1
    
    # Temperatura minima base
    tmin_base = 4.5 + seasonal_temp + warming_trend + altitude_effect + noise + chaotic
    
    # Temperatura maxima (con relacion no lineal con tmin)
    tmax_base = tmin_base + 10 + 2 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2.8, n_days)
    
    # Humedad (relacion no lineal)
    rh_base = 70 - 0.3 * tmin_base + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 14, n_days)
    rh_base = np.clip(rh_base, 25, 95)
    
    # Viento
    wind_base = 3.5 + np.random.normal(0, 2.2, n_days)
    wind_base = np.clip(wind_base, 0, 13)
    
    # Presion
    pressure_base = 101.3 * (1 - 0.0065 * elev / 288.15) ** 5.2561
    pressure_base += np.random.normal(0, 1.0, n_days)
    
    # Precipitacion (gamma)
    summer_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    prec_base = np.random.gamma(0.35, 1.8, n_days) * summer_factor
    prec_base = np.clip(prec_base, 0, 28)
    
    # Frost
    frost = (tmin_base <= 0).astype(int)
    
    df_station = pd.DataFrame({
        'station': station,
        'elevation': elev,
        'date': dates,
        'T2M_MIN': np.round(tmin_base, 2),
        'T2M_MAX': np.round(tmax_base, 2),
        'T2M_RANGE': np.round(tmax_base - tmin_base, 2),
        'RH2M': np.round(rh_base, 1),
        'WS2M': np.round(wind_base, 2),
        'PS': np.round(pressure_base, 2),
        'PRECTOTCORR': np.round(prec_base, 3),
        'frost': frost
    })
    
    all_data.append(df_station)

df = pd.concat(all_data, ignore_index=True)

# NO mezclar para mantener la estructura temporal
df.to_csv('datos_heladas_altiplano.csv', index=False)

print("="*50)
print("DATOS GENERADOS (CON CAOS Y NO LINEALIDAD)")
print("="*50)
print(f"Registros: {len(df)}")
print(f"Proporcion heladas: {df['frost'].mean():.2%}")
print(f"T2M_MIN media: {df['T2M_MIN'].mean():.2f}°C")
print(f"T2M_MIN std: {df['T2M_MIN'].std():.2f}°C")
print(f"Correlacion T2M_MAX vs T2M_MIN: {df['T2M_MAX'].corr(df['T2M_MIN']):.3f}")