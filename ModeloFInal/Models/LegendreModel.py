import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

# ===== CAPA LEGENDRE =====
class PolynomialLegendre(tf.keras.layers.Layer):
    def __init__(self, units, degree=2, use_bias=True, **kwargs):
        super(PolynomialLegendre, self).__init__(**kwargs)
        self.units = units
        self.degree = degree
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim_leg = input_shape[-1]

        self.kernel_leg = self.add_weight(
            shape=(input_dim_leg * self.degree, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name="kernel_leg"
        )

        if self.use_bias:
            self.bias_leg = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias_leg"
            )

    def call(self, inputs):
        # Convertimos la entrada al tipo de dato activo de la capa
        input_values = tf.cast(inputs, self.compute_dtype)

        # Valores iniciales de la recurrencia:
        # P0(x) = 1
        previous_previous_poly = tf.ones_like(input_values)

        # P1(x) = x
        previous_poly = input_values

        # Empezamos guardando el primer polinomio
        legendre_features = [previous_poly]

        # Generamos el resto de grados usando la recurrencia de Legendre
        for degree_index in range(2, self.degree + 1):
            current_degree = tf.cast(degree_index, self.compute_dtype)

            #Esto es el polinomio de Legendre pero calculado con la Recurrencia de Bonnet
            current_poly = ((2.0 * current_degree - 1.0) * input_values * previous_poly - (current_degree - 1.0) * previous_previous_poly) / current_degree

            legendre_features.append(current_poly)

            # SIguientes iteraciones
            previous_previous_poly = previous_poly
            previous_poly = current_poly

        # Unimos todas las bases polinómicas en un solo vector
        polynomial_basis = tf.concat(legendre_features, axis=-1)

        # Proyección lineal final, equivalente a una capa densa
        output_values = tf.matmul(polynomial_basis, self.kernel_leg)

        if self.use_bias:
            output_values = tf.nn.bias_add(output_values, self.bias_leg)

        return output_values

# ===== MODELO =====
def PolynomialDenseCreator_leg(degree_leg, input_dim_leg):
    inputPoli_leg = keras.Input(shape=(input_dim_leg,))
    x_leg = PolynomialLegendre(32, degree=degree_leg)(inputPoli_leg)
    x_leg = layers.Activation('swish')(x_leg)
    x_leg = layers.Dense(16, activation='swish')(x_leg)
    outputPoli_leg = layers.Dense(2, activation='softmax')(x_leg)
    
    model_leg = keras.Model(inputs=inputPoli_leg, outputs=outputPoli_leg)
    model_leg.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model_leg

def createEarlyStoppingCallback_leg(patience_leg=15):
    return keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience_leg,
        restore_best_weights=True
    )

def calculator_leg(scores_leg, num_splits):
    Totalloss_leg, Totalaccuracy_leg = 0, 0
    for loss, accuracy in scores_leg:
        Totalloss_leg += loss
        Totalaccuracy_leg += accuracy
    return Totalloss_leg/num_splits, Totalaccuracy_leg/num_splits

def save_results_to_csv_Legendre(scoreMean_dict, degrees_list, filename="temp_res_legendre.csv"):
    carpeta_destino = "resultados"
    os.makedirs(carpeta_destino, exist_ok=True)
    ruta_completa = os.path.join(carpeta_destino, filename)
    
    data = []
    for deg in degrees_list:
        data.append({
            "Polinomio": "Legendre",
            "Grado": deg,
            "Mejor_N": "N/A",
            "Loss_Promedio": round(scoreMean_dict[deg][0], 6),
            "Accuracy_Promedio": round(scoreMean_dict[deg][1], 6)
        })
    pd.DataFrame(data).to_csv(ruta_completa, index=False, sep=';')

# ===== EJECUCIÓN =====
magic_gamma_telescope_leg = fetch_ucirepo(id=159)
X_leg = magic_gamma_telescope_leg.data.features.to_numpy()
y_leg = magic_gamma_telescope_leg.data.targets.to_numpy()

epochs_leg = 120
num_splits_leg = 10
degrees = [3, 4, 5]

skf_leg = StratifiedKFold(n_splits=num_splits_leg, shuffle=True, random_state=1)
score_leg = {deg: [] for deg in degrees}

for train_index, test_index in skf_leg.split(X_leg, y_leg):
    X_train_leg, X_test_leg = X_leg[train_index], X_leg[test_index]
    y_train_leg, y_test_leg = y_leg[train_index], y_leg[test_index]

    scaler_leg = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled_leg = scaler_leg.fit_transform(X_train_leg)
    X_test_scaled_leg = scaler_leg.transform(X_test_leg)

    y_train_leg = (y_train_leg == 'g').astype(int)
    y_test_leg = (y_test_leg == 'g').astype(int)

    for deg in degrees:
        tf.keras.backend.clear_session()
        model = PolynomialDenseCreator_leg(deg, input_dim_leg=X_train_scaled_leg.shape[1])
        model.fit(X_train_scaled_leg, y_train_leg, validation_split=0.2, epochs=epochs_leg, batch_size=32, verbose=0, callbacks=[createEarlyStoppingCallback_leg()])
        eval_result = model.evaluate(X_test_scaled_leg, y_test_leg, verbose=0)
        score_leg[deg].append(eval_result)

scoreMean_leg = {deg: calculator_leg(score_leg[deg], num_splits_leg) for deg in degrees}

save_results_to_csv_Legendre(scoreMean_leg, degrees)