import multiprocessing
import subprocess
import os
import sys  # <--- Librería clave para la portabilidad

# 1. Limpieza de logs (opcional pero recomendado)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. DINAMISMO TOTAL: 
# sys.executable detecta automáticamente la ruta del Python que estés usando,
# ya sea en Windows, Linux, Mac o cualquier entorno de Conda.
PYTHON_EXE = sys.executable

# 3. RUTAS RELATIVAS:
# Usamos os.path.join para que las barras '/' o '\' se ajusten solas al sistema operativo.
scripts = [
    os.path.join("models", "ShymaliyModel.py"),
    os.path.join("models", "LegendreModel.py"),
    os.path.join("models", "ChebyshevModel.py"),
]

def run_script(script):
    if not os.path.exists(script):
        print(f"⚠️ No encuentro el archivo: {script}")
        print("Ruta revisada:", os.path.abspath(script))
        return

    print(f"🚀 Ejecutando: {script}")
    
    # Ejecución limpia y portable
    result = subprocess.run(
        [PYTHON_EXE, script],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"✅ Éxito: {script}")
    else:
        print(f"❌ Error en {script}:\n{result.stderr}")

if __name__ == "__main__":
    print(f"💻 Usando Python desde: {PYTHON_EXE}")
    
    num_procesos = min(len(scripts), multiprocessing.cpu_count())
    
    with multiprocessing.Pool(processes=num_procesos) as pool:
        pool.map(run_script, scripts)
        
    print("\n--- Proceso completado ---")