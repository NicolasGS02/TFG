# README - Ejecución del proyecto

Este repositorio incluye el código y el entorno necesario para ejecutar todos los modelos en **Python** de forma reproducible.

## 📁 Estructura relevante

Las rutas importantes dentro del proyecto son:

* `TFG/EntornoConda/Polynomial_Env.yml` → archivo del entorno Conda
* `TFG/codigo/main.py` → script principal para ejecutar todos los modelos en paralelo

---

## ⚙️ 1) Crear e instalar el entorno de trabajo

Antes de ejecutar el proyecto, es necesario instalar el entorno Conda incluido en el repositorio.

### Requisitos previos

* Tener instalado **Anaconda** o **Miniconda**
* Tener acceso a una terminal (`CMD`, PowerShell o Anaconda Prompt)

### Comando de instalación

Desde la raíz del repositorio, ejecuta:

```bash
conda env create -f TFG/EntornoConda/Polynomial_Env.yml
```

Esto creará automáticamente el entorno con todas las librerías y versiones necesarias.

---

## ▶️ 2) Activar el entorno

Una vez creado, activa el entorno con:

```bash
conda activate Polynomial_Env
```

> **Nota:** si el nombre del entorno definido dentro del archivo `.yml` es diferente, utiliza ese nombre en lugar de `Polynomial_Env`.

---

## 🚀 3) Ejecutar todos los modelos en paralelo

Para lanzar la ejecución completa de todos los modelos, debes:

1. Abrir una terminal **CMD**
2. Asegurarte de estar dentro del entorno `Polynomial_Env`
3. Ir a la carpeta `TFG\codigo`
4. Ejecutar el script principal con Python

### Comandos

```bash
cd TFG/codigo
python main.py
```

Este script se encarga de **ejecutar todos los modelos de forma paralela meidante hilos**, utilizando la configuración preparada en el entorno Conda.

---

## 💡 Ejecución recomendada (flujo completo)

Si quieres realizar todo el proceso seguido, estos serían los comandos completos:

```bash
conda env create -f TFG/EntornoConda/Polynomial_Env.yml
conda activate Polynomial_Env
cd TFG/codigo
python main.py
```
