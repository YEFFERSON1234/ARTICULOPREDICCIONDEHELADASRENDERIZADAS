import cdsapi
import os
import time

c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api", 
    key="b7dc4d7a-22b5-49e0-948b-ff80f94f96a1"
)

output_dir = 'datos_era5_puno'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Área ajustada más ceñida a Puno para reducir costos de datos
area_puno = [-13.9, -71.0, -17.5, -68.8]

# Definimos bloques de meses para no saturar el servidor
periodos = [
    {'nombre': 'sem1', 'meses': [f"{m:02d}" for m in range(1, 7)]},
    {'nombre': 'sem2', 'meses': [f"{m:02d}" for m in range(7, 13)]}
]

for year in range(2020, 2024):
    for p in periodos:
        filename = f"{output_dir}/era5_{year}_{p['nombre']}.nc"
        
        if os.path.exists(filename):
            continue

        print(f"-> Descargando: Año {year} - {p['nombre']}...")
        
        try:
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': [
                        '2m_temperature',
                        '2m_dewpoint_temperature',
                        'surface_pressure',
                        'total_precipitation',
                        'surface_solar_radiation_downwards'
                    ],
                    'year': str(year),
                    'month': p['meses'],
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                    'area': area_puno,
                    'data_format': 'netcdf',
                    'download_format': 'unarchived'
                },
                filename)
            print(f"   [OK] Guardado: {filename}")
        except Exception as e:
            print(f"   [!] Fallo en {year} {p['nombre']}: {e}")
        
        # Espera necesaria para la cola del servidor
        time.sleep(10)