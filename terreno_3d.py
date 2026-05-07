import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# 1. Cargar predicciones
df_pred = pd.read_csv('predictions.csv')
print(f"Predicciones cargadas: {len(df_pred)} registros")

# 2. Simular DEM (mientras consigues el real)
# Crear malla de coordenadas
latitudes = np.linspace(-17.5, -14.5, 100)  # Rango de Puno
longitudes = np.linspace(-71.5, -68.5, 100)
X, Y = np.meshgrid(longitudes, latitudes)

# Simular elevación (picos y valles)
Z = 3800 + 500 * np.sin(X*0.5) * np.cos(Y*0.5) + 200 * np.random.randn(*X.shape)
Z = np.clip(Z, 3700, 4800)

# 3. Crear heatmap de riesgo
# Interpolar predicciones en la malla
riesgo = np.random.rand(*X.shape)  # Simular (despues usas df_pred)

# 4. Grafico 3D
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Graficar superficie con color según riesgo
surf = ax.plot_surface(X, Y, Z, facecolors=cm.jet(riesgo), 
                        rstride=1, cstride=1, alpha=0.9, linewidth=0)

# Configuraciones
ax.set_xlabel('Longitud (°W)')
ax.set_ylabel('Latitud (°S)')
ax.set_zlabel('Elevación (m)')
ax.set_title('Mapa de Riesgo de Heladas - Altiplano Peruano')

# Barra de colores
mappable = cm.ScalarMappable(cmap='jet')
mappable.set_array(riesgo)
plt.colorbar(mappable, ax=ax, label='Probabilidad de Helada')

plt.savefig('mapa_3d_simple.png', dpi=150)
plt.show()
print("Mapa 3D generado")