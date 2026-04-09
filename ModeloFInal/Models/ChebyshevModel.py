import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# ===== CAPA CHEBYSHEV =====
class ChebyshevLayer(layers.Layer):
    def __init__(self, units, degree, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.degree = degree

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.degree + 1, input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="chebyshev_weights"
        )

    def call(self, inputs):
        previous_previous_poly = tf.ones_like(inputs)  # T0
        previous_poly = inputs                         # T1

        output_values = (
            tf.matmul(previous_previous_poly, self.w[0]) +
            tf.matmul(previous_poly, self.w[1])
        )

        for degree_index in range(2, self.degree + 1):
            current_poly = 2.0 * inputs * previous_poly - previous_previous_poly
            output_values += tf.matmul(current_poly, self.w[degree_index])

            previous_previous_poly = previous_poly
            previous_poly = current_poly

        return output_values


# ===== MODELO =====
def build_chebyshev_model(degree, input_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = ChebyshevLayer(32, degree=degree)(inputs)
    x = layers.Activation("swish")(x)
    x = layers.Dense(16, activation="swish")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ===== PLOT =====
def plot_cv_average_history_cheb(histories, degree, save_folder="resultados/imagenes"):
    os.makedirs(save_folder, exist_ok=True)

    max_epochs = max(len(h.history["loss"]) for h in histories)
    epochs = np.arange(1, max_epochs + 1)

    def get_metric(metric):
        matrix = np.full((len(histories), max_epochs), np.nan)
        for i, h in enumerate(histories):
            values = h.history[metric]
            matrix[i, :len(values)] = values
        return np.nanmean(matrix, axis=0)

    avg_loss = get_metric("loss")
    avg_val_loss = get_metric("val_loss")
    avg_acc = get_metric("accuracy")
    avg_val_acc = get_metric("val_accuracy")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_loss)
    plt.plot(epochs, avg_val_loss)
    plt.title(f"Pérdida Promedio - Chebyshev G{degree}")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_acc)
    plt.plot(epochs, avg_val_acc)
    plt.title(f"Accuracy Promedio - Chebyshev G{degree}")

    plt.tight_layout()
    file_path = os.path.join(save_folder, f"chebyshev_grado_{degree}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


# ===== CSV =====
def save_results_cheb( score_mean, degrees, filename="temp_res_chebyshev.csv"):
    os.makedirs("resultados", exist_ok=True)

    data = []
    for deg in degrees:
        data.append({
            "Polinomio": "Chebyshev",
            "Grado": deg,
            "Mejor_N": "N/A",
            "Loss_Promedio": round(score_mean[deg][0], 6),
            "Accuracy_Promedio": round(score_mean[deg][1], 6)
        })

    pd.DataFrame(data).to_csv(
        os.path.join("resultados", filename),
        index=False,
        sep=";"
    )


# ===== EJECUCIÓN =====
dataset = fetch_ucirepo(id=159)
X = dataset.data.features.to_numpy()
y = dataset.data.targets.to_numpy()

degrees = [1,2,3,4,5]
epochs = 120
num_splits = 10

skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1)

scores = {deg: [] for deg in degrees}
histories = {deg: [] for deg in degrees}

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
 
    y_train = (y_train == 'g').astype(int)
    y_test = (y_test == 'g').astype(int)

    # =========================
    # 2) Validación interna estratificada
    # =========================
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.3,
        stratify=y_train,
        random_state=42
    )


    # =========================
    # 3) Normalización
    # =========================
    scaler = MinMaxScaler(feature_range=(-1, 1))

    X_subtrain = scaler.fit_transform(X_subtrain)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    for deg in degrees:
        tf.keras.backend.clear_session()

        model = build_chebyshev_model(deg, X_subtrain.shape[1])

        history = model.fit(
            X_subtrain,
            y_subtrain,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

        score = model.evaluate(X_test, y_test, verbose=0)

        histories[deg].append(history)
        scores[deg].append(score)

score_mean = {
    deg: (
        np.mean([x[0] for x in scores[deg]]),
        np.mean([x[1] for x in scores[deg]])
    )
    for deg in degrees
}

save_results_cheb(score_mean, degrees)

for deg in degrees:
    plot_cv_average_history_cheb(histories[deg], deg)