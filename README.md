# README - Ejecución del Proyecto

Este repositorio incluye el código y el entorno necesario para ejecutar los modelos de redes neuronales polinómicas en **Python** de forma reproducible.

## 📁 Estructura del Proyecto

Las rutas y archivos principales dentro del repositorio son:


* `EntornoConda/Polynomial_Env.yml` → Archivo de configuración del entorno Conda.
* `main.py` → Script principal para ejecutar la experimentación con los modelos.
* `models.py` → Definición de los diferentes modelos y validación cruzada.
* `data_utils.py` → Utilidades para la carga y gestión de conjuntos de datos.
* `metrics_utils.py` → Funciones de cálculo de métricas de rendimiento.
* `plot_utils.py` → Utilidades para la generación de gráficas y visualización.

## ⚙️ 1) Crear e instalar el entorno de trabajo

Antes de ejecutar el proyecto, es necesario instalar el entorno Conda incluido.

### Requisitos previos

* Tener instalado **Anaconda** o **Miniconda**.
* Acceso a una terminal (`CMD`, PowerShell o Anaconda Prompt).

### Comando de instalación

Desde la raíz del repositorio, ejecuta:


conda env create -f EntornoConda/Polynomial_Env.yml

Esto creará automáticamente el entorno con todas las dependencias y versiones necesarias.

---

## ▶️ 2) Activar el entorno

Una vez creado, activa el entorno con:

```bash
conda activate Polynomial_Env
```

---

## 🚀 3) Ejecutar el proyecto

Para lanzar la ejecución de los modelos:

```bash
1. Abre tu terminal y asegúrate de estar dentro del entorno `Polynomial_Env`.
2. Sitúate en la raíz del repositorio.
3. Ejecuta el script principal con Python:

python main.py
```

El script se encarga de procesar los conjuntos de datos, entrenar los diferentes modelos (Lineal, Chebyshev, Legendre y Shmaliy) utilizando validación cruzada y guardar los resultados y gráficas generadas.

---

## 💡 Flujo de ejecución completo

Si deseas realizar todo el proceso desde cero:

```bash
conda env create -f EntornoConda/Polynomial_Env.yml
conda activate Polynomial_Env
python main.py
```
