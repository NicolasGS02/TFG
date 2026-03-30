# Librerías
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np

class ShmaliyLayer(layers.Layer):
    def __init__(self, units, degree, N=100, **kwargs):
        super(ShmaliyLayer, self).__init__(**kwargs)
        self.units = units
        self.degree = degree
        self.N = float(N) # Parámetro de longitud de datos de Shmaliy

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.degree + 1, input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="shmaliy_weights"
        )

    def call(self, inputs):
        # Llevamos la entrada desde [-1, 1] al rango discreto [0, N-1]
        discrete_positions = (inputs + 1.0) * (self.N - 1.0) / 2.0

        # Primeros valores de la recurrencia
        # S0 = 1
        previous_previous_poly = tf.ones_like(discrete_positions)

        # S1 = 1 - (2k)/(N-1)
        previous_poly = 1.0 - (2.0 * discrete_positions) / (self.N - 1.0)

        # Salida inicial con los dos primeros grados
        output_values = (
            tf.matmul(previous_previous_poly, self.w[0]) +
            tf.matmul(previous_poly, self.w[1])
        )

        # Y aqui de forma recursiva al tener los primeros calculamos el resto de términos hasta el grado deseado
        for degree_index in range(1, self.degree):
            denominator = (degree_index + 1.0) * (self.N - 1.0 - degree_index)

            recurrence_term = ((2.0 * degree_index + 1.0) * (self.N - 1.0 - 2.0 * discrete_positions)  * previous_poly) / denominator

            correction_term = (degree_index * (self.N + degree_index)* previous_previous_poly) / denominator

            current_poly = recurrence_term - correction_term

            # Añadimos la contribución del nuevo grado
            output_values += tf.matmul(current_poly, self.w[degree_index + 1])

            # Actualizamos
            previous_previous_poly = previous_poly
            previous_poly = current_poly

        return output_values
# ===== MODELO =====
def PolynomialDenseCreator_Shm(degree_Shm,nValues_Shm,input_dim_Shm):
    inputPoli_Shm = keras.Input(shape=(input_dim_Shm,))
    
    x_Shm = ShmaliyLayer(32, degree=degree_Shm, N=nValues_Shm)(inputPoli_Shm)
    x_Shm = layers.Activation('swish')(x_Shm)
    x_Shm = layers.Dense(16, activation='swish')(x_Shm)
    
    outputPoli_Shm = layers.Dense(2, activation='softmax')(x_Shm)
    
    model_Shm = keras.Model(
        inputs=inputPoli_Shm,
        outputs=outputPoli_Shm,
        name=f"Polynomial_Model_Degree_{degree_Shm}_Shm"
    )
    
    model_Shm.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model_Shm

def createEarlyStoppingCallback_Shm(patience_Shm=15):
    return keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience_Shm,
        restore_best_weights=True
    )

def calculator_Shm(scores_Shm):
    Totalloss_Shm, Totalaccuracy_Shm = 0, 0
    for loss, accuracy in scores_Shm:
        Totalloss_Shm += loss
        Totalaccuracy_Shm += accuracy

    return Totalloss_Shm/num_splits_Shm, Totalaccuracy_Shm/num_splits_Shm

def save_results_to_csv_Shm(scoreMean_dict, search_results_dict, degrees_list, filename="temp_res_shmaliy.csv"):
    """
    Guarda los resultados en modeloFinal/resultados/temp_res_shmaliy.csv
    """
    # 1. Definir la ruta de la carpeta (un nivel arriba '..', carpeta 'resultados')
    carpeta_destino = "resultados"
    
    # 2. Asegurarse de que la carpeta exista (si no existe, la crea sin dar error)
    os.makedirs(carpeta_destino, exist_ok=True)
    
    # 3. Crear la ruta completa del archivo
    ruta_completa = os.path.join(carpeta_destino, filename)
    
    data = []
    
    for deg in degrees_list:
        best_n = max(search_results_dict[deg], key=search_results_dict[deg].get)
        avg_loss = scoreMean_dict[deg][0]
        avg_accuracy = scoreMean_dict[deg][1]
        
        data.append({
            "Polinomio": "Shmaliy",
            "Grado": deg,
            "Mejor_N": best_n,
            "Loss_Promedio": round(avg_loss, 6),
            "Accuracy_Promedio": round(avg_accuracy, 6)
        })
        
    df_resultados = pd.DataFrame(data)
    
    # 4. Guardar el archivo en la ruta específica
    df_resultados.to_csv(ruta_completa, index=False, sep=';')


# --- 3. CONFIGURACIÓN DE LA BÚSQUEDA ---
from sklearn.model_selection import StratifiedKFold

# ===== DATOS =====
magic_gamma_telescope_Shm = fetch_ucirepo(id=159)

X_Shm = magic_gamma_telescope_Shm.data.features 
y_Shm = magic_gamma_telescope_Shm.data.targets 


# ===== HIPERPARÁMETROS =====
epochs_Shm = 120
batch_size_Shm = 32
input_dim_Shm = X_Shm.shape[1]
num_splits_Shm = 10
degrees = [3, 4, 5]



N_candidates = [25, 50, 100, 250, 500] # Valores de N a probar
n_splits = 2 # Usamos 2 folds para no alargar demasiado la búsqueda
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

# Diccionario para almacenar resultados: {grado: {N: accuracy_promedio}}
search_results = {d: {} for d in degrees}

# Asegurar datos en numpy
X_Shm = X_Shm.to_numpy() if hasattr(X_Shm, 'to_numpy') else X_Shm
y_Shm = y_Shm.to_numpy() if hasattr(y_Shm, 'to_numpy') else y_Shm

# --- 4. EJECUCIÓN DE LA BÚSQUEDA (GRID SEARCH) ---
for deg in degrees:
    
    for n_val in N_candidates:
        fold_accs = []
        
        for train_idx, test_idx in skf.split(X_Shm, y_Shm):
            # Limpieza de memoria
            tf.keras.backend.clear_session()
            
            # Split y Normalización
            X_train, X_test = X_Shm[train_idx], X_Shm[test_idx]
            y_train, y_test = y_Shm[train_idx], y_Shm[test_idx]
            
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            y_train_bin = (y_train == 'g').astype(int)
            y_test_bin = (y_test == 'g').astype(int)
            
            # Modelo
            model = PolynomialDenseCreator_Shm(deg, n_val, X_Shm.shape[1])
            
            # Entrenamiento rápido para búsqueda
            model.fit(
                X_train_scaled, y_train_bin,
                epochs=10, # Menos épocas para la fase de búsqueda
                batch_size=64,
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                validation_split=0.1
            )
            
            # Evaluación
            _, acc = model.evaluate(X_test_scaled, y_test_bin, verbose=0)
            fold_accs.append(acc)
            
        avg_acc = np.mean(fold_accs)
        search_results[deg][n_val] = avg_acc

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

skf_Shm = StratifiedKFold(n_splits=num_splits_Shm, shuffle=True, random_state=1)

history_Shm = {deg: [] for deg in degrees}
score_Shm = {deg: [] for deg in degrees}


for train_index, test_index in tqdm(skf_Shm.split(X_Shm, y_Shm), total=num_splits_Shm):

    X_train_Shm, X_test_Shm = X_Shm[train_index], X_Shm[test_index]
    y_train_Shm, y_test_Shm = y_Shm[train_index], y_Shm[test_index]

    scaler_Shm = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled_Shm = scaler_Shm.fit_transform(X_train_Shm)
    X_test_scaled_Shm = scaler_Shm.transform(X_test_Shm)

    y_train_Shm = (y_train_Shm == 'g').astype(int)
    y_test_Shm = (y_test_Shm == 'g').astype(int)


    for deg in degrees:
        tf.keras.backend.clear_session()
        best_n = max(search_results[deg], key=search_results[deg].get)
        best_val = search_results[deg][best_n]
        model = PolynomialDenseCreator_Shm(deg, best_n, input_dim_Shm=X_train_scaled_Shm.shape[1])
        TranedModel = model.fit(X_train_scaled_Shm, y_train_Shm, validation_split=0.2, epochs=epochs_Shm, batch_size=32, verbose=0)
        eval_result = model.evaluate(X_test_scaled_Shm, y_test_Shm, verbose=0)
        
        history_Shm[deg].append(TranedModel)
        score_Shm[deg].append(eval_result)


scoreMean_shm = {}
for deg in degrees:
    scoreMean_shm[deg] = calculator_Shm(score_Shm[deg])


# Guardar resultados en CSV
import pandas as pd
import os


save_results_to_csv_Shm(scoreMean_shm, search_results, degrees, filename="temp_res_shmaliy.csv")