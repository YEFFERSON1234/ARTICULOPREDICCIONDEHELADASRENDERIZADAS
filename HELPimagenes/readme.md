leer : new, organizacion

data/dem_puno_completo.tif  ←  utils/unificar_dems.py (si tienes varios DEM)
         ↓
data/datos_heladas_altiplano.csv  ←  Tus datos históricos
         ↓
models/random_forest_model.py     ←  Entrena el modelo
         ↓
outputs/predictions.csv           ←  Predicciones guardadas
         ↓
visualization/mapa_predicciones.py  ←  ESTE SCRIPT
         ├─ Carga DEM:   utils/visualizacion_utils.cargar_y_preparar_dem()
         ├─ Carga pred:   utils/visualizacion_utils.cargar_predictions_csv()
         ├─ Interpola:    utils/visualizacion_utils.mapear_prediccion_a_dem()
         └─ Grafica:      utils/visualizacion_utils.crear_mapa_heladas()
              ↓
         outputs/mapa_heladas_3d_realista.png  ←  ¡Para tu artículo!