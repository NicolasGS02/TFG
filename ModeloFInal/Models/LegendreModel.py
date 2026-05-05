# ===== LIBRERÍAS =====
import time
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== CAPA LEGENDRE =====
class PolynomialLegendre(tf.keras.layers.Layer):
    def __init__(self, units, degree=2, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.degree = degree
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim * self.degree, self.units),
            initializer="glorot_uniform",
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True
            )

    def call(self, inputs):
        x = tf.cast(inputs, self.compute_dtype)

        prev_prev = tf.ones_like(x)   # P0
        prev = x                      # P1

        features = [prev]

        for n in range(2, self.degree + 1):
            n = tf.cast(n, self.compute_dtype)
            current = ((2*n - 1)*x*prev - (n - 1)*prev_prev) / n

            features.append(current)

            prev_prev = prev
            prev = current

        basis = tf.concat(features, axis=-1)

        output = tf.matmul(basis, self.kernel)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        return output

# ===== MODELO =====
def PolynomialDenseCreator_leg(degree, input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))

    x = PolynomialLegendre(32, degree=degree)(inputs)
    x = layers.Activation('swish')(x)
    x = layers.Dense(16, activation='relu')(x)

    output = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, output)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ===== CALLBACK =====
def createEarlyStopping(patience=15):
    return keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

# ===== MÉTRICAS =====
def calculator(scores, times):
    loss = np.mean([s[0] for s in scores])
    acc = np.mean([s[1] for s in scores])
    t = np.mean(times)
    return loss, acc, t

# ===== PLOT =====
def save_image_plot(histories, degree, save_folder="resultados/imagenes"):
    os.makedirs(save_folder, exist_ok=True)

    max_epochs = max(len(h.history['loss']) for h in histories)
    epochs = np.arange(1, max_epochs + 1)

    def get_metric(metric):
        matrix = np.full((len(histories), max_epochs), np.nan)
        for i, h in enumerate(histories):
            values = h.history[metric]
            matrix[i, :len(values)] = values
        return np.nanmean(matrix, axis=0)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, get_metric("loss"))
    plt.plot(epochs, get_metric("val_loss"))
    plt.title(f"Loss - Grado {degree}")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, get_metric("accuracy"))
    plt.plot(epochs, get_metric("val_accuracy"))
    plt.title(f"Accuracy - Grado {degree}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"legendre_grado_{degree}.png"))
    plt.close()

# ===== DATOS =====
idDataset = 53  # Cambia aquí
dataset = fetch_ucirepo(id=idDataset)

X = dataset.data.features.to_numpy()
y = dataset.data.targets.to_numpy()

# ===== HIPERPARÁMETROS =====
epochs = 400
batch_size = 32
num_splits = 10
degrees = [2, 3, 4, 5, 6]

# ===== CROSS VALIDATION =====
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1)

history = {deg: [] for deg in degrees}
scores = {deg: [] for deg in degrees}
times = {deg: [] for deg in degrees}

for train_idx, test_idx in skf.split(X, y):

    # Split
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train.ravel())
    y_test = le.transform(y_test.ravel())

    # Validación interna
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train,
        test_size=0.3,
        stratify=y_train,
        random_state=42
    )

    # Normalización
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_sub = scaler.fit_transform(X_sub)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    num_classes = len(np.unique(y_train))
    input_dim = X_sub.shape[1]

    for deg in degrees:
        tf.keras.backend.clear_session()

        model = PolynomialDenseCreator_leg(deg, input_dim, num_classes)

        start = time.time()

        hist = model.fit(
            X_sub, y_sub,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

        end = time.time()

        result = model.evaluate(X_test, y_test, verbose=0)

        history[deg].append(hist)
        scores[deg].append(result)
        times[deg].append(end - start)

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
        "Polinomio": "Legendre",
        "Grado": deg,
        "Mejor_N": "N/A",
        "Loss_Promedio": round(loss, 8),
        "Accuracy_Promedio": round(acc, 8),
        "Tiempo_Promedio(s)": round(t, 2)
    })

pd.DataFrame(data).to_csv("resultados/resultados_legendre.csv", index=False, sep=';')

# ===== GRÁFICAS =====
for deg in degrees:
    save_image_plot(history[deg], deg)

print("Proceso finalizado correctamente")