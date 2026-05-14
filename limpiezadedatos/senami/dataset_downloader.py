import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import Select

# Configuración del navegador
options = webdriver.ChromeOptions()
# options.add_argument('--headless') # Descomenta para correr sin ver la ventana
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def login_senamhi(user, password):
    driver.get("https://www.senamhi.gob.pe/servicios/?p=descarga-datos-meteorologicos")
    # Aquí el script debe buscar los campos de login si el portal los solicita
    # Nota: A veces el acceso es directo tras sesión activa en el navegador
    print("Por favor, inicia sesión manualmente en la ventana abierta y presiona Enter aquí.")
    input()

def descargar_estacion(id_estacion, anio_inicio, anio_fin):
    for anio in range(anio_inicio, anio_fin + 1):
        # 1. Seleccionar Estación
        # 2. Seleccionar Rango de Fechas (Enero a Diciembre del 'anio')
        # 3. Hacer clic en el botón 'Descargar' o 'Consultar'
        print(f"Procesando año: {anio} para la estación {id_estacion}")
        
        # Lógica de scraping específica para los selectores del SENAMHI
        # Ejemplo de selección de año:
        # select_anio = Select(driver.find_element(By.ID, "id_del_selector_anio"))
        # select_anio.select_by_visible_text(str(anio))
        
        time.sleep(2) # Pausa para evitar bloqueos del servidor

# Ejecución
login_senamhi("fewok51146@ellbit.com", "dcc0v4q")
descargar_estacion("PUNO_001", 2000, 2023)