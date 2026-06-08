# 📥 Guía para Replicar en Otra Computadora

## 🎯 Resumen

Esta guía te permite clonar el proyecto y replicarlo completamente en otra máquina.

---

## 🚀 Pasos para Replicar

### **1️⃣ Clonar el Repositorio**

```bash
# HTTPS (recomendado si no tienes SSH configurado)
git clone https://github.com/YEFFERSON1234/ARTICULOPREDICCIONDEHELADASRENDERIZADAS.git

# O SSH (si tienes clave SSH configurada)
git clone git@github.com:YEFFERSON1234/ARTICULOPREDICCIONDEHELADASRENDERIZADAS.git

# Entrar a la carpeta
cd ARTICULOPREDICCIONDEHELADASRENDERIZADAS
```

### **2️⃣ Verificar que se Clonó Todo**

```bash
# Ver archivos principales
ls -la  # En Linux/Mac
dir     # En Windows PowerShell

# Archivos clave que deben existir:
# ✓ README.md
# ✓ requirements.txt
# ✓ config.py
# ✓ main_pipeline.py
# ✓ docs/ (carpeta con toda la documentación)
# ✓ data_process/ (con datos históricos)
# ✓ modelos/ (con 11 modelos de ML)
```

### **3️⃣ Crear Entorno Virtual**

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### **4️⃣ Instalar Dependencias**

```bash
# Opción 1: Desde requirements.txt (RECOMENDADO)
pip install -r requirements.txt

# Opción 2: Ver docs/configuracion/setup.md para instalación manual
```

### **5️⃣ Verificar Instalación Rápidamente**

```bash
# Ejecutar verificación rápida
python quickstart.py

# Esto verifica:
# ✓ Python version (debe ser 3.11+)
# ✓ Imports de paquetes
# ✓ Archivos de datos históricos
# ✓ Directorios de modelos
# ✓ Sistema de visualización
# ✓ Tests básicos
```

### **6️⃣ Ejecutar el Pipeline Completo**

```bash
# Pipeline automatizado (5 pasos)
python main_pipeline.py

# O con opciones:
python main_pipeline.py --verbose      # Con detalles
python main_pipeline.py --quiet        # Sin salida
python main_pipeline.py --predict-only # Solo predicción
python main_pipeline.py --visualize-only # Solo visualización
```

### **7️⃣ Visualizar Resultados**

```bash
# Opción 1: Mapa estático (siempre funciona)
python visualization/unified_visualizer.py --mode static

# Opción 2: 3D interactivo (si OpenGL disponible)
python visualization/unified_visualizer.py --mode interactive

# Opción 3: Animación temporal
python visualization/unified_visualizer.py --mode animated

# Opción 4: Mapa de riesgo de helada
python visualization/unified_visualizer.py --mode risk
```

### **8️⃣ (Opcional) Ejecutar Tests**

```bash
# Todos los tests
python -m pytest tests/ -v

# Solo tests de datos
python -m pytest tests/test_data_preparation.py -v

# Solo tests específicos
python -m pytest tests/test_data_preparation.py::TestDataIntegrity -v
```

---

## 📋 Checklist de Replicación

- [ ] Clonaste el repositorio
- [ ] Creaste y activaste el entorno virtual
- [ ] Instalaste todas las dependencias
- [ ] Ejecutaste `quickstart.py` con ✓
- [ ] Ejecutaste `main_pipeline.py` sin errores
- [ ] Visualizaste los resultados
- [ ] (Opcional) Ejecutaste los tests

---

## 🆘 Problemas Comunes

### **"No such file or directory: data_process/datos_heladas_puno_REAL.csv"**
→ Verifica que `data_process/` contenga los archivos CSV necesarios

### **"ModuleNotFoundError"**
→ Ejecuta: `pip install -r requirements.txt --upgrade`

### **"OpenGL not available"**
→ Usa visualización estática: `python visualization/unified_visualizer.py --mode static`

### **"CUDA not available"**
→ Normal si no tienes GPU. El sistema usa CPU automáticamente.

---

## 📞 Soporte

Consulta:
- [setup.md](setup.md) - Instalación detallada
- [docs/arquitectura/](../arquitectura/) - Cómo funciona el sistema
- [docs/datos-modelos/](../datos-modelos/) - Resultados y métricas

---

**Último update:** 2026-06-07