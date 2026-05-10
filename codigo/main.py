import os
import sys
import time
import multiprocessing
from datetime import datetime

# Evitar mensajes en la terminal irrelevante
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd

from data_utils import list_csv_files, load_dataset
from models import (
    run_lineal_cv,
    run_chebyshev_cv,
    run_legendre_cv,
    run_shmaliy_cv,
)
from plot_utils import plot_history


# ===== PARAMETROS PRINCIPALES =====

# Rutas
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "data_uci")
RESULTS_PATH = os.path.join(BASE_DIR, "resultados")
IMAGES_PATH = os.path.join(BASE_DIR, "imagenes")

# Hiperparametros
N_FOLDS = 10
EPOCHS = 400 #Como se tiene el early stop, se puede dejar un numero alto para asegurar que se alcance la convergencia
BATCH_SIZE = 32
PATIENCE = 20

# Grados a estudiar
CHEBYSHEV_DEGREES = [2, 3, 4, 5, 6]
LEGENDRE_DEGREES = [2, 3, 4, 5, 6]
SHMALIY_DEGREES = [2, 3, 4, 5, 6]
SHMALIY_N_CANDIDATES = [25, 50, 100, 250]


def main(data_path=DATA_PATH):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(IMAGES_PATH, exist_ok=True)

    timestamp = datetime.now().strftime("%y%m%d_%H%M_")
    dataset_files = list_csv_files(data_path)

    print(f"Entorno Python: {sys.executable}")
    print(f"Datos: {data_path}")
    print("Modelos: 4 (Lineal, Chebyshev, Legendre, Shmaliy)")
    print(f"Procesos: {min(4, multiprocessing.cpu_count())}")
    print("Inicio de ejecucion...\n")

    global_metrics = []
    global_times = {}

    for file_path in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        X, y = load_dataset(file_path)

        args = (
            X,
            y,
            N_FOLDS,
            EPOCHS,
            BATCH_SIZE,
            PATIENCE,
            CHEBYSHEV_DEGREES,
            LEGENDRE_DEGREES,
            SHMALIY_DEGREES,
            SHMALIY_N_CANDIDATES,
            dataset_name,
            timestamp,
            IMAGES_PATH,
        )

        model_names = ["lineal", "chebyshev", "legendre", "shmaliy"]
        with multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count())) as pool:
            tasks = [(name, args) for name in model_names]
            results = pool.starmap(run_model_by_name, tasks)

        lineal_res, cheb_res, leg_res, shm_res = results

        add_global_metrics(global_metrics, dataset_name, lineal_res)
        add_global_metrics(global_metrics, dataset_name, cheb_res)
        add_global_metrics(global_metrics, dataset_name, leg_res)
        add_global_metrics(global_metrics, dataset_name, shm_res)

        add_global_times(global_times, dataset_name, lineal_res)
        add_global_times(global_times, dataset_name, cheb_res)
        add_global_times(global_times, dataset_name, leg_res)
        add_global_times(global_times, dataset_name, shm_res)

        print(f"Finalizado: {dataset_name}")

    save_global_files(timestamp, global_metrics, global_times, RESULTS_PATH)
    print("\n--- Proceso completado ---")


def run_model_by_name(model_name, args):
    (
        X,
        y,
        n_folds,
        epochs,
        batch_size,
        patience,
        cheb_degrees,
        leg_degrees,
        shm_degrees,
        shm_n_candidates,
        dataset_name,
        timestamp,
        images_path,
    ) = args

    if model_name == "lineal":
        return run_lineal(X, y, n_folds, epochs, batch_size, patience, dataset_name, timestamp, images_path)
    if model_name == "chebyshev":
        return run_chebyshev(X, y, n_folds, epochs, batch_size, patience, cheb_degrees, dataset_name, timestamp, images_path)
    if model_name == "legendre":
        return run_legendre(X, y, n_folds, epochs, batch_size, patience, leg_degrees, dataset_name, timestamp, images_path)
    return run_shmaliy(X, y, n_folds, epochs, batch_size, patience, shm_degrees, shm_n_candidates, dataset_name, timestamp, images_path)


def run_lineal(X, y, n_folds, epochs, batch_size, patience, dataset_name, timestamp, images_path):
    metrics, histories, avg_time = run_lineal_cv(X, y, n_folds, epochs, batch_size, patience)
    save_plots(timestamp, dataset_name, "lineal", {"N/A": histories}, images_path)
    return {
        "model": "Lineal",
        "metrics": {"N/A": metrics},
        "times": {"N/A": avg_time},
    }


def run_chebyshev(X, y, n_folds, epochs, batch_size, patience, degrees, dataset_name, timestamp, images_path):
    metrics, histories, avg_times = run_chebyshev_cv(X, y, n_folds, degrees, epochs, batch_size, patience)
    save_plots(timestamp, dataset_name, "chebyshev", histories, images_path)
    return {
        "model": "Chebyshev",
        "metrics": metrics,
        "times": avg_times,
    }


def run_legendre(X, y, n_folds, epochs, batch_size, patience, degrees, dataset_name, timestamp, images_path):
    metrics, histories, avg_times = run_legendre_cv(X, y, n_folds, degrees, epochs, batch_size, patience)
    save_plots(timestamp, dataset_name, "legendre", histories, images_path)
    return {
        "model": "Legendre",
        "metrics": metrics,
        "times": avg_times,
    }


def run_shmaliy(X, y, n_folds, epochs, batch_size, patience, degrees, n_candidates, dataset_name, timestamp, images_path):
    metrics, histories, avg_times, search_results = run_shmaliy_cv(X, y, n_folds, degrees, n_candidates, epochs, batch_size, patience)
    save_plots(timestamp, dataset_name, "shmaliy", histories, images_path)
    return {
        "model": "Shmaliy",
        "metrics": metrics,
        "times": avg_times,
        "best_n": {deg: max(search_results[deg], key=search_results[deg].get) for deg in degrees},
    }


def save_plots(prefix, dataset_name, model_key, histories, images_path):
    dataset_folder = os.path.join(images_path, f"{prefix}{dataset_name}")
    os.makedirs(dataset_folder, exist_ok=True)

    for degree, history_list in histories.items():
        degree_str = str(degree) if degree != "N/A" else "NA"
        title = f"{dataset_name} - {model_key} - {degree}"
        file_name = f"{prefix}{dataset_name}_{model_key}_{degree_str}.png"
        file_path = os.path.join(dataset_folder, file_name)
        plot_history(history_list, title, file_path)


def add_global_metrics(global_metrics, dataset_name, result):
    best_n = result.get("best_n", {})
    for degree, metrics in result["metrics"].items():
        global_metrics.append({
            "Dataset": dataset_name,
            "Modelo": result["model"],
            "Grado": degree,
            "Mejor_N": best_n.get(degree, "N/A"),
            "Accuracy": metrics.get("accuracy"),
            "F1_score": metrics.get("f1_score"),
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "Tiempo_Promedio(s)": result["times"].get(degree),
        })


def add_global_times(global_times, dataset_name, result):
    if dataset_name not in global_times:
        global_times[dataset_name] = {"Dataset": dataset_name}

    model_times = list(result["times"].values())
    if model_times:
        model_total = float(sum(model_times))
        global_times[dataset_name][result["model"]] = round(model_total, 3)


def metric_prefix(model_name, degree):
    if degree == "N/A":
        return f"{model_name}_NA"
    return f"{model_name}_G{degree}"


def save_global_files(prefix, global_metrics, global_times, results_path):
    metrics_path = os.path.join(results_path, f"{prefix}global_result.xlsx")
    times_path = os.path.join(results_path, f"{prefix}global_time.xlsx")

    pd.DataFrame(global_metrics).to_excel(metrics_path, index=False)
    pd.DataFrame(list(global_times.values())).to_excel(times_path, index=False)


if __name__ == "__main__":
    main()
