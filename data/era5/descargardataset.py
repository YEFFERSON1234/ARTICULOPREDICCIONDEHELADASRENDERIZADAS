import cdsapi
import os
import sys
import time

# CDS API credentials must be set via environment variables or ~/.cdsapirc.
# See .env.example for the required variable names.
cds_url = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
cds_key = os.environ.get("CDS_API_KEY")
if not cds_key:
    print("ERROR: Set the CDS_API_KEY environment variable.")
    print("See .env.example for details.")
    sys.exit(1)

c = cdsapi.Client(url=cds_url, key=cds_key)

output_dir = 'datos_era5_puno'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Área ajustada más ceñida a Puno para reducir costos de datos
area_puno = [-13.9, -71.0, -17.5, -68.8]

# Bucle principal por año (2020 a 2023)
for year in range(2015, 2018):
    # Nuevo bucle secundario para recorrer los 12 meses del año
    for month in range(1, 13):
        # Convertimos el número de mes a texto con dos dígitos (ej: '01', '02', ..., '12')
        month_str = f"{month:02d}"
        
        # El nombre del archivo ahora incluye el año y el mes específico
        filename = f"{output_dir}/era5_{year}_{month_str}.nc"
        
        # Si el mes ya se descargó previamente, lo salta
        if os.path.exists(filename):
            continue

        print(f"-> Descargando: Año {year} - Mes {month_str}...")
        
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
                    'month': month_str,  # <--- Enviamos solo el mes actual en el bucle
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                    'area': area_puno,
                    'data_format': 'netcdf',
                    'download_format': 'unarchived'
                },
                filename)
            print(f"   [OK] Guardado: {filename}")
        except Exception as e:
            print(f"   [!] Fallo en {year} - Mes {month_str}: {e}")
        
        # Pausa crucial para la cola del servidor de Copernicus
        time.sleep(10)