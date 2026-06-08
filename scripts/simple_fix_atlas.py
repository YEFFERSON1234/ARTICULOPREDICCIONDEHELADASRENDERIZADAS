import pandas as pd
import numpy as np

try:
    # Leer el archivo
    print("Leyendo archivo...")
    df = pd.read_csv('data_process/CSV_MAESTRO_CONSOLIDADO.csv')
    print(f"Archivo leído: {len(df)} registros")
    
    # Función para calcular nivel de riesgo
    def calcular_nivel_riesgo(prob):
        if pd.isna(prob):
            return 'Desconocido'
        elif prob < 0.2:
            return 'Muy Bajo'
        elif prob < 0.4:
            return 'Bajo'
        elif prob < 0.6:
            return 'Moderado'
        elif prob < 0.8:
            return 'Alto'
        else:
            return 'Muy Alto'
    
    # Agregar columna nivel_riesgo basada en prob_ensemble_promedio
    if 'prob_ensemble_promedio' in df.columns:
        df['nivel_riesgo'] = df['prob_ensemble_promedio'].apply(calcular_nivel_riesgo)
    else:
        # Usar probabilidad_helada si no existe ensemble
        df['nivel_riesgo'] = df['probabilidad_helada'].apply(calcular_nivel_riesgo)
    
    # Guardar
    print("Guardando archivo...")
    df.to_csv('data_process/final_frost_atlas.csv', index=False)
    print("✅ Archivo final_frost_atlas.csv generado exitosamente")
    
    # Mostrar estadísticas
    print(f"Total de registros: {len(df)}")
    print(f"Distribución de niveles de riesgo:")
    print(df['nivel_riesgo'].value_counts())
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()