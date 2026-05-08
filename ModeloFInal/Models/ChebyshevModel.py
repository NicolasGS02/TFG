# ===== LIBRERÍAS =====
import time
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import gc

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
        # T0 = 1
        prev_prev = tf.ones_like(inputs)

        # T1 = x
        prev = inputs

        output = (
            tf.matmul(prev_prev, self.w[0]) +
            tf.matmul(prev, self.w[1])
        )

        # Recurrencia Chebyshev
        for n in range(2, self.degree + 1):
            current = 2.0 * inputs * prev - prev_prev
            output += tf.matmul(current, self.w[n])

            prev_prev = prev
            prev = current

        return output


# ===== MODELO =====
def build_chebyshev_model(degree, input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))

    x = ChebyshevLayer(64, degree=degree)(inputs)
    x = layers.Activation("swish")(x)
    x = layers.Dense(16, activation="relu")(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ===== PLOT =====
def save_image_plot(histories, degree, save_folder="resultados/imagenes"):
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


# ===== MÉTRICAS =====
def calculator(scores, times):
    loss = np.mean([s[0] for s in scores])
    acc = np.mean([s[1] for s in scores])
    t = np.mean(times)
    return loss, acc, t


# ===== EARLY STOPPING =====
def create_early_stopping(patience=15):
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )


# ===== DATOS =====
idDataset = 159
dataset = fetch_ucirepo(id=idDataset)

X = dataset.data.features.to_numpy()
y = dataset.data.targets.to_numpy()


# ===== HIPERPARÁMETROS =====
degrees = [2, 3, 4, 5, 6]
epochs = 400
num_splits = 10

skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1)

scores = {deg: [] for deg in degrees}
histories = {deg: [] for deg in degrees}
times = {deg: [] for deg in degrees}


# ===== CROSS VALIDATION =====
for train_idx, test_idx in skf.split(X, y):

    # Split
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Codificación multiclase
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Validación interna
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.3,
        stratify=y_train,
        random_state=42
    )

    # Normalización
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_subtrain = scaler.fit_transform(X_subtrain)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Nº clases dinámico
    num_classes = len(np.unique(y_train))

    # Entrenamiento por grado
    for deg in degrees:
        tf.keras.backend.clear_session()

        model = build_chebyshev_model(deg, X_subtrain.shape[1], num_classes)

        start = time.time()

        history = model.fit(
            X_subtrain,
            y_subtrain,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=[create_early_stopping()]
        )

        end = time.time()

        score = model.evaluate(X_test, y_test, verbose=0)

        histories[deg].append(history)
        scores[deg].append(score)
        times[deg].append(end - start)

        del model
        gc.collect()


# ===== RESULTADOS =====
results = {}
for deg in degrees:
    results[deg] = calculator(scores[deg], times[deg])

# ===== GUARDADO CSV =====
os.makedirs("resultados", exist_ok=True)

data = []
for deg in degrees:
    loss, acc, t = results[deg]

    data.append({
        "Polinomio": "Chebyshev",
        "Grado": deg,
        "Mejor_N": "N/A",
        "Loss_Promedio": round(loss, 8),
        "Accuracy_Promedio": round(acc, 8),
        "Tiempo_Promedio(s)": round(t, 2)
    })

pd.DataFrame(data).to_csv("resultados/resultados_chebyshev.csv", index=False, sep=';')


# ===== GRÁFICAS =====
for deg in degrees:
    save_image_plot(histories[deg], deg)

print("Proceso finalizado correctamente")
