# 📚 Documentación del Proyecto - Predicción de Heladas en Puno

Bienvenido a la documentación centralizada del proyecto de predicción de heladas en el Altiplano Peruano.

## 🗂️ Estructura Organizada

Esta documentación está organizada en **4 secciones temáticas principales** para una navegación clara y eficiente:

---

## 1️⃣ 🏗️ [ARQUITECTURA](arquitectura/) 
**Documentación técnica y diseño del sistema**

Comprende la estructura, componentes y flujo de datos del sistema:

- [arquitectura.txt](arquitectura/arquitectura.txt) - Diagrama ASCII de la arquitectura general
- [INFORME_PIPELINE_COMPLETO.md](arquitectura/INFORME_PIPELINE_COMPLETO.md) - **Descripción detallada del pipeline (6 pasos completos)**
- [DOCUMENTACION_GRAFICOS.md](arquitectura/DOCUMENTACION_GRAFICOS.md) - Documentación de 10+ gráficos generados
- [articulo_completo.tex](arquitectura/articulo_completo.tex) - Artículo académico en LaTeX

**Ideal para:** Entender cómo funciona el sistema, ver el pipeline completo, y conocer la arquitectura.

---

## 2️⃣ 📊 [DATOS Y MODELOS](datos-modelos/)
**Información sobre datos, modelos y predicciones**

Documentación sobre los datasets, modelos de ML y resultados:

- [README_CSV_MAESTRO_PREDICCIONES.md](datos-modelos/README_CSV_MAESTRO_PREDICCIONES.md) - **Descripción del CSV maestro consolidado (73,170 registros)**
- [METRICAS_MODELOS_DETALLADO.md](datos-modelos/METRICAS_MODELOS_DETALLADO.md) - Métricas de 11 modelos diferentes
- [INFORME_GENERAL_CONSOLIDADO.txt](datos-modelos/INFORME_GENERAL_CONSOLIDADO.txt) - Resumen consolidado
- [INFORME_GENERAL_RESULTADOS.md](datos-modelos/INFORME_GENERAL_RESULTADOS.md) - Resultados detallados

**Ideal para:** Ver métricas de modelos, entender los datos, analizar predicciones.

---

## 3️⃣ ⚙️ [CONFIGURACIÓN](configuracion/)
**Dependencias, versiones y configuración del entorno**

Guía de instalación y requisitos del sistema:

- [package_versions.txt](configuracion/package_versions.txt) - **Versiones exactas de todos los paquetes (PyTorch 2.5.1, CUDA 12.1, XGBoost 3.2.0, etc.)**
- [download_srtm.txt](configuracion/download_srtm.txt) - Instrucciones para descargar tiles SRTM de NASA
- [unify_dem.txt](configuracion/unify_dem.txt) - Explicación del proceso de unificación de DEM

**Ideal para:** Configurar el entorno, instalar dependencias, descargar datos geográficos.

---

## 4️⃣ 🔧 [MANTENIMIENTO](mantenimiento/)
**Mejoras implementadas e items pendientes**

Información sobre el estado del proyecto y áreas de mejora:

- [IMPROVEMENTS.md](mantenimiento/IMPROVEMENTS.md) - **Mejoras de ingeniería de software implementadas**
- [MISSING_ITEMS.md](mantenimiento/MISSING_ITEMS.md) - Análisis de 20+ faltantes y priorización

**Ideal para:** Ver qué se ha mejorado, identificar trabajo pendiente, planificar próximas fases.

---

## 📍 GUÍAS DE INICIO RÁPIDO

### Para empezar rápidamente:
1. Leer [SETUP.md](SETUP.md) - Instrucciones de instalación
2. Leer [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) - Cómo ejecutar el proyecto
3. Leer [REPLICATE.md](REPLICATE.md) - Cómo reproducir en otra máquina

### Para entender el proyecto:
1. Ver [arquitectura/INFORME_PIPELINE_COMPLETO.md](arquitectura/INFORME_PIPELINE_COMPLETO.md)
2. Consultar [datos-modelos/README_CSV_MAESTRO_PREDICCIONES.md](datos-modelos/README_CSV_MAESTRO_PREDICCIONES.md)
3. Revisar [datos-modelos/METRICAS_MODELOS_DETALLADO.md](datos-modelos/METRICAS_MODELOS_DETALLADO.md)

---

## 🎯 Características Principales

✅ **11 modelos de ML** - XGBoost, Random Forest, SVM, MLP, LSTM, CNN-1D, SARIMAX, Prophet, Holt-Winters, SARIMA-ANN, Ensemble Maestro

✅ **Excelente rendimiento** - RMSE 2.42°C, F1-Score 0.963, AUC-ROC 0.996

✅ **29 estaciones** - Monitoreadas desde 2002-2023 (392,283 registros)

✅ **Visualización 3D** - OpenGL interactivo con terreno renderizado

✅ **Predicción flexible** - Para cualquier fecha específica, no solo el día siguiente

✅ **Ensemble maestro** - Combina 4 mejores modelos con pesos optimizados

---

## 🔗 Recursos Adicionales

### Prototipos y Multimedia
- **[prototipos/](prototipos/)** - Prototipo de visualización 3D web
- **[media/](media/)** - Animaciones y screenshots del proyecto

### Archivos Principales de Ejecución
- **[SETUP.md](SETUP.md)** - Setup completo
- **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - Guía de ejecución
- **[REPLICATE.md](REPLICATE.md)** - Cómo reproducir en otra computadora

---

## 📈 Resumen de Contenido

| Sección | Archivos | Propósito |
|---------|----------|----------|
| **Arquitectura** | 4 | Entender cómo funciona el sistema |
| **Datos & Modelos** | 4 | Analizar resultados y métricas |
| **Configuración** | 3 | Instalar y configurar el entorno |
| **Mantenimiento** | 2 | Ver mejoras e items pendientes |
| **Prototipos** | varios | Visualizaciones 3D |
| **Multimedia** | varios | Gráficos y animaciones |

---

## 💡 Consejos de Navegación

- **Si acabas de empezar:** Empieza por [SETUP.md](SETUP.md) y [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)
- **Si quieres entender el sistema:** Lee [arquitectura/INFORME_PIPELINE_COMPLETO.md](arquitectura/INFORME_PIPELINE_COMPLETO.md)
- **Si quieres analizar modelos:** Consulta [datos-modelos/METRICAS_MODELOS_DETALLADO.md](datos-modelos/METRICAS_MODELOS_DETALLADO.md)
- **Si necesitas configurable:** Ve a [configuracion/package_versions.txt](configuracion/package_versions.txt)
- **Si buscas mejoras futuras:** Revisa [mantenimiento/MISSING_ITEMS.md](mantenimiento/MISSING_ITEMS.md)

---

**Última actualización:** 2026-06-07  
**Versión:** 1.0 - Documentación Unificada y Organizada
