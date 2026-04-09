import multiprocessing
import subprocess
import os
import sys
import time

# 1. Limpieza de logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
PYTHON_EXE = sys.executable

# 2. RUTAS RELATIVAS
scripts = [
    os.path.join("models", "ShymaliyModel.py"),
    os.path.join("models", "LegendreModel.py"),
    os.path.join("models", "ChebyshevModel.py"),
    os.path.join("models", "LinealModel.py"),
]

def run_script(script):
    if not os.path.exists(script):
        print(f"No encuentro el archivo: {script}")
        return (script, 0, "No encontrado")

    print(f"Ejecutando: {script} ...")
    
    # Inicio del cronómetro
    inicio = time.time()
    
    result = subprocess.run(
        [PYTHON_EXE, script],
        capture_output=True,
        text=True
    )
    
    #Fin del cronómetro
    fin = time.time()
    tiempo_total = fin - inicio

    if result.returncode == 0:
        print(f"Éxito: {script}")
        return (script, tiempo_total, "Éxito")
    else:
        print(f"Error en {script}:\n{result.stderr}")
        return (script, tiempo_total, "Error")

if __name__ == "__main__":
    print(f"Entorno Virtual: {PYTHON_EXE}")
    print(f"Iniciando ejecución paralela en {multiprocessing.cpu_count()} núcleos...\n")
    
    num_procesos = min(len(scripts), multiprocessing.cpu_count())
    
    # Ejecutamos y guardamos los retornos en 'resultados'
    with multiprocessing.Pool(processes=num_procesos) as pool:
        resultados = pool.map(run_script, scripts)
        
    #
    print("\n" + "="*60)
    print(f"{'ARCHIVO':<30} | {'ESTADO':<12} | {'TIEMPO'}")
    print("-" * 60)
    
    for script, tiempo, estado in resultados:
        # Formatear tiempo a min y seg si es muy largo
        if tiempo > 60:
            t_str = f"{int(tiempo//60)}m {tiempo%60:.2f}s"
        else:
            t_str = f"{tiempo:.2f}s"
            
        nombre_fichero = os.path.basename(script)
        print(f"{nombre_fichero:<30} | {estado:<12} | {t_str}")
        
    print("="*60)
    print("--- Proceso completado ---")