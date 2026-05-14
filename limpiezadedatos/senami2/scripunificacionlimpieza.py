import pandas as pd
import glob
import os

# Configuración de rutas
input_path = "limpiezadedatos/senami2/nombresdatos semani/"
output_file = "limpiezadedatos/datos_heladas_puno_REAL.csv"

# Diccionario Maestro: Coordenadas y Zonas del Departamento de Puno
coords_puno = {
    # --- ZONA NORTE (Melgar, Azángaro, Carabaya, Huancané, Moho) ---
    "ANANEA": {"lat": -14.68, "lon": -69.53, "zona": "Norte"},
    "ARAPA": {"lat": -15.14, "lon": -70.12, "zona": "Norte"},
    "AYAVIRI": {"lat": -14.88, "lon": -70.59, "zona": "Norte"},
    "AZANGARO": {"lat": -14.91, "lon": -70.19, "zona": "Norte"},
    "CHUQUIBAMBILLA": {"lat": -14.79, "lon": -70.72, "zona": "Norte"},
    "COJATA": {"lat": -15.02, "lon": -69.49, "zona": "Norte"},
    "CRUCERO": {"lat": -14.36, "lon": -70.02, "zona": "Norte"},
    "CUYO CUYO": {"lat": -14.50, "lon": -69.54, "zona": "Norte"},
    "HUANCANE": {"lat": -15.21, "lon": -69.76, "zona": "Norte"},
    "HUARAYA MOHO": {"lat": -15.36, "lon": -69.50, "zona": "Norte"},
    "MUÑANI": {"lat": -14.77, "lon": -69.95, "zona": "Norte"},
    "PROGRESO": {"lat": -14.69, "lon": -70.35, "zona": "Norte"},
    "PUCARA": {"lat": -15.04, "lon": -70.36, "zona": "Norte"},
    "PUTINA": {"lat": -14.91, "lon": -69.87, "zona": "Norte"},
    "SANTA ROSA": {"lat": -14.61, "lon": -70.78, "zona": "Norte"},

    # --- ZONA CENTRO (Puno, Lampa, San Román) ---
    "CABANILLAS": {"lat": -15.64, "lon": -70.35, "zona": "Centro"},
    "CAPACHICA": {"lat": -15.64, "lon": -69.83, "zona": "Centro"},
    "CRUCERO ALTO": {"lat": -15.77, "lon": -71.01, "zona": "Centro"},
    "LAMPA": {"lat": -15.36, "lon": -70.36, "zona": "Centro"},
    "PAMPAHUTA": {"lat": -15.54, "lon": -70.32, "zona": "Centro"},
    "PUNO": {"lat": -15.84, "lon": -70.02, "zona": "Centro"},

    # --- ZONA SUR (Chucuito, El Collao, Yunguyo) ---
    "CAPAZO": {"lat": -17.17, "lon": -69.75, "zona": "Sur"},
    "DESAGUADERO": {"lat": -16.56, "lon": -69.04, "zona": "Sur"},
    "ILAVE": {"lat": -16.08, "lon": -69.46, "zona": "Sur"},
    "ISLA SUANA": {"lat": -16.35, "lon": -68.88, "zona": "Sur"},
    "ISLA TAQUILE": {"lat": -15.77, "lon": -69.68, "zona": "Sur"},
    "JULI": {"lat": -16.21, "lon": -69.46, "zona": "Sur"},
    "MAZO CRUZ": {"lat": -16.75, "lon": -69.71, "zona": "Sur"},
    "PIZACOMA": {"lat": -16.91, "lon": -69.36, "zona": "Sur"},
    "TAHUACO - YUNGUYO": {"lat": -16.24, "lon": -69.09, "zona": "Sur"}
}

def unificar_departamental():
    archivos = glob.glob(os.path.join(input_path, "*.txt"))
    data_total = []

    print(f"Iniciando unificación de {len(archivos)} estaciones del departamento...")

    for f in archivos:
        # Extraer nombre limpio (ej: ANANEA)
        estacion = os.path.basename(f).replace(".txt", "").upper()
        try:
            # Lectura según tutorial SENAMHI: delimitado por espacios [cite: 23]
            df = pd.read_csv(f, sep='\s+', header=None, 
                             names=['year', 'month', 'day', 'precip', 'tmax', 'tmin'])
            
            # Limpiar nulos (-99.9) [cite: 23]
            df = df.replace(-99.9, pd.NA)
            
            # Asignar metadatos geográficos exactos
            info = coords_puno.get(estacion, {})
            df['estacion'] = estacion
            df['lat'] = info.get('lat', pd.NA)
            df['lon'] = info.get('lon', pd.NA)
            df['zona'] = info.get('zona', 'No Clasificada')
            df['departamento'] = 'PUNO'
            
            # Crear índice temporal real
            df['fecha'] = pd.to_datetime(df[['year', 'month', 'day']])
            data_total.append(df)
            print(f"✓ Procesada: {estacion} ({info.get('zona', 'N/A')})")
            
        except Exception as e:
            print(f"✗ Error en {estacion}: {e}")

    if data_total:
        df_final = pd.concat(data_total).dropna(subset=['tmin'])
        df_final.to_csv(output_file, index=False)
        print(f"\n--- Resumen ---")
        print(f"Archivo generado: {output_file}")
        print(f"Registros totales: {len(df_final)}")
        print(f"Zonas cubiertas: {df_final['zona'].unique()}")

if __name__ == "__main__":
    unificar_departamental()