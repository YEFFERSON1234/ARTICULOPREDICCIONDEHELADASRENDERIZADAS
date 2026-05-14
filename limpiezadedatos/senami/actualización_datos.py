import os
import csv

# 1. Configuración de rutas dinámicas
directorio_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_entrada = os.path.join(directorio_actual, 'dataset')
carpeta_salida = os.path.join(directorio_actual, 'dataset-limpio')
anio_limite = 2000

# Crear la carpeta de salida si no existe
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# 2. Procesamiento
if os.path.exists(carpeta_entrada):
    archivos = [f for f in os.listdir(carpeta_entrada) if f.endswith('.txt')]
    
    for nombre_archivo in archivos:
        ruta_completa = os.path.join(carpeta_entrada, nombre_archivo)
        datos_filtrados = []
        
        # Leer el TXT original
        with open(ruta_completa, 'r', encoding='latin-1') as archivo:
            for linea in archivo:
                columnas = linea.split() # Divide por espacios
                if columnas:
                    try:
                        if int(columnas[0]) >= anio_limite:
                            datos_filtrados.append(columnas)
                    except ValueError:
                        continue
        
        # 3. Guardar como CSV
        if datos_filtrados:
            # Cambiamos la extensión .txt por .csv
            nombre_csv = nombre_archivo.replace('.txt', '.csv')
            ruta_guardado = os.path.join(carpeta_salida, nombre_csv)
            
            with open(ruta_guardado, 'w', newline='', encoding='utf-8') as f_csv:
                # Usamos delimiter=',' para comas, o ';' si tu Excel es en español
                escritor = csv.writer(f_csv, delimiter=',')
                
                # Opcional: Agregar encabezados si sabes qué significa cada columna
                # escritor.writerow(['Anio', 'Mes', 'Dia', 'Valor1', 'Valor2', 'Valor3'])
                
                escritor.writerows(datos_filtrados)
                
            print(f"Convertido: {nombre_archivo} -> {nombre_csv} ({len(datos_filtrados)} filas)")
else:
    print(f"No se encontró la carpeta: {carpeta_entrada}")

print("\n--- Proceso finalizado ---")