# 📥 Guía para Replicar en Otra Computadora

## 🎯 Resumen de la Subida

✅ **Todos los cambios han sido subidos a GitHub** en el commit:
- **Hash**: `bcde1f3`
- **Rama**: `main`
- **Archivos**: 10 nuevos (1,947 líneas de código)

---

## 🚀 Pasos para Replicar en Otra Computadora

### **1️⃣ Clonar el Repositorio**

```bash
# HTTPS
git clone https://github.com/YEFFERSON1234/ARTICULOPREDICCIONDEHELADASRENDERIZADAS.git

# O SSH (si tienes configurado)
git clone git@github.com:YEFFERSON1234/ARTICULOPREDICCIONDEHELADASRENDERIZADAS.git

# Entrar a la carpeta
cd ARTICULOPREDICCIONDEHELADASRENDERIZADAS
```

### **2️⃣ Verificar que se Clonó Todo**

```bash
# Ver archivos nuevos
ls -la  # En Linux/Mac
dir     # En Windows PowerShell

# Verificar que existen los 10 archivos nuevos:
# ✓ config.py
# ✓ logging_config.py
# ✓ main_pipeline.py
# ✓ quickstart.py
# ✓ consolidation_plan.py
# ✓ IMPROVEMENTS.md
# ✓ SETUP.md
# ✓ visualization/unified_visualizer.py
# ✓ tests/test_data_preparation.py
# ✓ tests/__init__.py
```

### **3️⃣ Verificar Instalación Rápidamente**

```bash
# Ejecutar verificación
python quickstart.py

# Esto verifica:
# ✓ Python version
# ✓ Imports correctos
# ✓ Archivos de datos
# ✓ Directorio de modelos
# ✓ Visualización
# ✓ Tests básicos
```

### **4️⃣ Ejecutar el Pipeline Completo**

```bash
# Pipeline automatizado (5 pasos)
python main_pipeline.py

# O con opciones:
python main_pipeline.py --verbose    # Con detalles
python main_pipeline.py --predict-only  # Solo predicción
python main_pipeline.py --visualize-only # Solo visualización
```

### **5️⃣ Visualizar Resultados**

```bash
# Opción 1: Mapa estático (siempre funciona)
python visualization/unified_visualizer.py --mode static

# Opción 2: 3D interactivo (si OpenGL disponible)
python visualization/unified_visualizer.py --mode interactive

# Opción 3: Animación temporal
python visualization/unified_visualizer.py --mode animated

# Opción 4: Mapa de riesgo
python visualization/unified_visualizer.py --mode risk
```

### **6️⃣ Ejecutar Tests (Opcional)**

```bash
# Todos los tests
python -m pytest tests/ -v

# Solo tests de datos
python -m pytest tests/test_data_preparation.py -v
```

---

## 📋 Checklist para la Nueva Computadora

- [ ] Git instalado (`git --version`)
- [ ] Python 3.8+ instalado (`python --version`)
- [ ] `requirements.txt` actualizado y dependencias instaladas
- [ ] Clonar repositorio
- [ ] Ejecutar `python quickstart.py`
- [ ] Ejecutar `python main_pipeline.py`
- [ ] Verificar que se crean: `logs/`, `models/`
- [ ] Probar visualización

---

## 📁 Estructura que Tendrás Después de Clonar

```
ARTICULOPREDICCIONDEHELADASRENDERIZADAS/
├── config.py                      ✨ Importable
├── logging_config.py              ✨ Importable
├── main_pipeline.py               ✨ Script principal
├── quickstart.py                  ✨ Verificación
├── consolidation_plan.py          ✨ Análisis
├── IMPROVEMENTS.md                ✨ Documentación
├── SETUP.md                       ✨ Guía
├── REPLICATE.md                   ← Este archivo
│
├── visualization/
│   ├── unified_visualizer.py      ✨ Nuevo
│   ├── main.py
│   └── ...
│
├── tests/                         ✨ Nueva carpeta
│   ├── __init__.py
│   ├── test_data_preparation.py
│
├── logs/                          (Se crea automáticamente)
├── models/                        (Se crea automáticamente)
│
├── data/
├── data_process/
├── src/
├── modelos/
└── ... (todos los archivos originales)
```

---

## 🔧 Troubleshooting en Nueva Computadora

| Problema | Solución |
|----------|----------|
| `git: command not found` | Instala Git desde https://git-scm.com |
| `ModuleNotFoundError: No module named 'config'` | Asegúrate de estar en directorio raíz |
| `requirements not satisfied` | `pip install -r requirements.txt` |
| `No SENAMHI file found` | Verifica que `data_process/datos_heladas_puno_REAL.csv` existe |
| OpenGL error | Usa `--mode static` en lugar de `--mode interactive` |

---

## ✅ Verificación de Sincronización

Para verificar que tienes todo sincronizado:

```bash
# Ver commit más reciente
git log -1

# Ver rama actual
git branch

# Ver archivos nuevos
git log --oneline | head -5

# Ver cambios de este commit
git show bcde1f3 --stat
```

---

## 🎯 Resumen Rápido (Copy-Paste)

```bash
# 1. Clonar
git clone https://github.com/YEFFERSON1234/ARTICULOPREDICCIONDEHELADASRENDERIZADAS.git
cd ARTICULOPREDICCIONDEHELADASRENDERIZADAS

# 2. Verificar
python quickstart.py

# 3. Ejecutar
python main_pipeline.py

# 4. Visualizar
python visualization/unified_visualizer.py --mode static

# ¡Listo!
```

---

## 📞 Si Algo Falla

1. Revisar logs: `cat logs/frost_prediction.log`
2. Ejecutar tests: `python -m pytest tests/ -v`
3. Ver commit: `git show bcde1f3`
4. Leer: `SETUP.md` y `IMPROVEMENTS.md`

---

**Versión**: 1.0  
**Fecha de Subida**: 2 de junio de 2026  
**Commit**: bcde1f3  
**Archivos**: 10 nuevos (1,947 líneas)  
**Estado**: ✅ TODO SUBIDO Y SINCRONIZADO
