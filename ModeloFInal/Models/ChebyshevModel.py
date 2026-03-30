import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

# ===== CAPA CHEBYSHEV =====
class ChebyshevLayer(layers.Layer):
    def __init__(self, units, degree, **kwargs):
        super(ChebyshevLayer, self).__init__(**kwargs)
        self.units = units
        self.degree = degree

    def build(self, input_shape):
        # Pesos para cada grado del polinomio
        self.w = self.add_weight(
            shape=(self.degree + 1, input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="chebyshev_weights"
        )

    def call(self, inputs):
        # Valores iniciales de la recurrencia de Chebyshev
        # T0(x) = 1
        previous_previous_poly = tf.ones_like(inputs)

        # T1(x) = x
        previous_poly = inputs

        # Salida inicial con los dos primeros grados
        output_values = (
            tf.matmul(previous_previous_poly, self.w[0]) +
            tf.matmul(previous_poly, self.w[1])
        )

        # Aqui de forma recursiva generamos los otros grados 
        for degree_index in range(2, self.degree + 1):
            current_poly = 2.0 * inputs * previous_poly - previous_previous_poly

            output_values += tf.matmul(current_poly, self.w[degree_index])

            # Preparamos la siguiente iteración
            previous_previous_poly = previous_poly
            previous_poly = current_poly

        return output_values

# ===== MODELO =====
def PolynomialDenseCreator_Cheb(degree_Cheb, input_dim_Cheb):
    inputPoli_Cheb = keras.Input(shape=(input_dim_Cheb,))
    x_Cheb = ChebyshevLayer(32, degree=degree_Cheb)(inputPoli_Cheb)
    x_Cheb = layers.Activation('swish')(x_Cheb)
    x_Cheb = layers.Dense(16, activation='swish')(x_Cheb)
    outputPoli_Cheb = layers.Dense(2, activation='softmax')(x_Cheb)
    
    model_Cheb = keras.Model(inputs=inputPoli_Cheb, outputs=outputPoli_Cheb)
    model_Cheb.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model_Cheb

def calculator_Cheb(scores_Cheb, num_splits):
    Totalloss_Cheb, Totalaccuracy_Cheb = 0, 0
    for loss, accuracy in scores_Cheb:
        Totalloss_Cheb += loss
        Totalaccuracy_Cheb += accuracy
    return Totalloss_Cheb/num_splits, Totalaccuracy_Cheb/num_splits

def save_results_to_csv_Chebyshev(scoreMean_dict, degrees_list, filename="temp_res_chebyshev.csv"):
    carpeta_destino = "resultados"
    os.makedirs(carpeta_destino, exist_ok=True)
    ruta_completa = os.path.join(carpeta_destino, filename)
    
    data = []
    for deg in degrees_list:
        data.append({
            "Polinomio": "Chebyshev",
            "Grado": deg,
            "Mejor_N": "N/A",
            "Loss_Promedio": round(scoreMean_dict[deg][0], 6),
            "Accuracy_Promedio": round(scoreMean_dict[deg][1], 6)
        })
    df_resultados = pd.DataFrame(data)
    df_resultados.to_csv(ruta_completa, index=False, sep=';')

# ===== PROCESAMIENTO =====
magic_gamma_telescope_Cheb = fetch_ucirepo(id=159)
X_Cheb = magic_gamma_telescope_Cheb.data.features.to_numpy()
y_Cheb = magic_gamma_telescope_Cheb.data.targets.to_numpy()

num_splits_Cheb = 10
epochs_Cheb = 120
degrees = [3, 4, 5]
skf_Cheb = StratifiedKFold(n_splits=num_splits_Cheb, shuffle=True, random_state=1)
score_Cheb = {deg: [] for deg in degrees}

for train_index, test_index in skf_Cheb.split(X_Cheb, y_Cheb):
    X_train_Cheb, X_test_Cheb = X_Cheb[train_index], X_Cheb[test_index]
    y_train_Cheb, y_test_Cheb = y_Cheb[train_index], y_Cheb[test_index]

    scaler_Cheb = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled_Cheb = scaler_Cheb.fit_transform(X_train_Cheb)
    X_test_scaled_Cheb = scaler_Cheb.transform(X_test_Cheb)

    y_train_Cheb = (y_train_Cheb == 'g').astype(int)
    y_test_Cheb = (y_test_Cheb == 'g').astype(int)

    for deg in degrees:
        tf.keras.backend.clear_session()
        model = PolynomialDenseCreator_Cheb(deg, input_dim_Cheb=X_train_scaled_Cheb.shape[1])
        model.fit(X_train_scaled_Cheb, y_train_Cheb, validation_split=0.2, epochs=epochs_Cheb, batch_size=32, verbose=0)
        eval_result = model.evaluate(X_test_scaled_Cheb, y_test_Cheb, verbose=0)
        score_Cheb[deg].append(eval_result)

# Cálculo de promedios finales
scoreMean_Cheb = {deg: calculator_Cheb(score_Cheb[deg], num_splits_Cheb) for deg in degrees}

# Guardado final
save_results_to_csv_Chebyshev(scoreMean_Cheb, degrees)