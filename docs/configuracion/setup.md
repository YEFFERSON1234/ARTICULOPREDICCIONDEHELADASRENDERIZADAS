# 🎯 Setup Completo - Instalación y Configuración

## ✅ Requisitos Previos

- Python 3.11.9
- Windows PowerShell o Terminal bash
- 4GB RAM mínimo
- Conexión a Internet (para descargas iniciales)

---

## 🛠️ Instalación Completa

### **Paso 1: Crear Entorno Virtual**

```powershell
# Crear entorno virtual
python -m venv venv

# Activar (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activar (bash/Linux/Mac)
source venv/bin/activate
```

### **Paso 2: Actualizar pip**

```powershell
pip install --upgrade pip
```

### **Paso 3: Instalar PyTorch (CUDA 12.1)**

```powershell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### **Paso 4: Instalar Dependencias de Datos**

```powershell
# NumPy, Pandas, SciPy
pip install numpy==2.4.3 pandas==3.0.2 scipy==1.17.1

# Machine Learning
pip install xgboost==3.2.0 scikit-learn==1.8.0

# Geoespacial (para archivos GeoTIFF)
pip install rasterio==1.4.4 affine==2.4.0
```

### **Paso 5: Instalar Visualización**

```powershell
# Matplotlib y Seaborn
pip install matplotlib==3.10.9 seaborn==0.13.2

# OpenCV y visión
pip install opencv-python==4.13.0.92 pillow==12.1.1

# OpenGL para visualización 3D
pip install pyopengl glfw
```

### **Paso 6: Instalar Dependencias Adicionales**

```powershell
# Utilidades
pip install joblib==1.5.3 networkx==3.6.1 sympy==1.13.1 jinja2==3.1.6

# Geoespacial avanzado
pip install xarray geemap

# Opcional: Google Earth Engine (requiere autenticación)
pip install earthengine-api
```

### **Paso 7: Instalar desde requirements.txt**

```powershell
# O instalar todo de una vez desde requirements.txt
pip install -r requirements.txt
```

---

## ✅ Verificar Instalación

```powershell
# Ver versión de Python
python --version

# Ver versión de PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Listar paquetes instalados
pip list
```

---

## 📂 Estructura de Directorios Necesarios

Asegúrate de que existan:

```
ARTICULOPREDICCIONDEHELADASRENDERIZADAS/
├── data/
│   ├── datos_era5_puno/
│   ├── datos_senami_puno/
│   └── (archivos .nc y .txt)
├── data_process/
│   ├── datos_heladas_puno_REAL.csv
│   └── (archivos procesados)
├── modelos/
│   └── (archivos .py de modelos)
├── Archivos.tiff-renderizar/
│   └── (archivos .tif del DEM)
└── Render/Mapa/
    └── (scripts de renderizado)
```

Si falta alguno, créalos:

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force -Path data/datos_era5_puno
New-Item -ItemType Directory -Force -Path data_process
New-Item -ItemType Directory -Force -Path modelos
```

---

## 🔍 Troubleshooting

### **Error: "No module named 'torch'"**
```powershell
# Reinstalar PyTorch
pip uninstall torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### **Error: "CUDA no disponible"**
```powershell
# Esto es normal si no tienes GPU NVIDIA
# El sistema caerá a CPU automáticamente
python -c "import torch; print(torch.cuda.is_available())"
```

### **Error: "ModuleNotFoundError"**
```powershell
# Reinstalar todas las dependencias
pip install -r requirements.txt --force-reinstall
```

### **Problema con OpenGL**
```powershell
# Si hay problemas de visualización 3D:
pip install pyopengl pyopengl-accelerate
```

---

## 🚀 Verificación Rápida

Una vez instalado, ejecuta:

```powershell
python quickstart.py
```

Esto verifica:
- ✓ Versión de Python
- ✓ Imports de módulos
- ✓ Archivos de datos
- ✓ Directorio de modelos
- ✓ Sistema de visualización
- ✓ Tests básicos

Si todo está ✓, ¡estás listo para usar el proyecto!