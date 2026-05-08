# ===== LIBRERÍAS =====
import time
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
import gc


# ===== MODELO =====
def create_model(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(16, activation='relu')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='LinearModel')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ===== PLOT =====
def plot_cv_average_history(histories, save_folder="resultados/imagenes"):
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
    plt.title("Pérdida Promedio - Modelo Lineal")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_acc)
    plt.plot(epochs, avg_val_acc)
    plt.title("Accuracy Promedio - Modelo Lineal")

    plt.tight_layout()
    file_path = os.path.join(save_folder, "lineal.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


# ===== EARLY STOPPING =====
def create_early_stopping(patience=15):
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )


# ===== DATOS =====
dataset = fetch_ucirepo(id=159)

X = dataset.data.features.to_numpy()
y = dataset.data.targets.to_numpy()


# ===== HIPERPARÁMETROS =====
epochs = 400
num_splits = 10

skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1)

scores = []
histories = []
times = []


# ===== CROSS VALIDATION =====
for train_idx, test_idx in skf.split(X, y):

    # =========================
    # 1) Split
    # =========================
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # =========================
    # 2) Label encoding (MULTICLASE)
    # =========================
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # =========================
    # 3) Validación interna
    # =========================
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.3,
        stratify=y_train,
        random_state=42
    )

    # =========================
    # 4) Normalización
    # =========================
    scaler = MinMaxScaler(feature_range=(-1, 1))

    X_subtrain = scaler.fit_transform(X_subtrain)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # =========================
    # 5) Modelo
    # =========================
    num_classes = len(np.unique(y_train))

    tf.keras.backend.clear_session()
    model = create_model(X_subtrain.shape[1], num_classes)

    # =========================
    # 6) Entrenamiento
    # =========================
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

    # =========================
    # 7) Evaluación
    # =========================
    score = model.evaluate(X_test, y_test, verbose=0)

    histories.append(history)
    scores.append(score)
    times.append(end - start)

    del model
    gc.collect()


# ===== RESULTADOS =====
def calculator(scores, times):
    total_loss, total_acc, total_time = 0, 0, 0

    for (loss, acc), t in zip(scores, times):
        total_loss += loss
        total_acc += acc
        total_time += t

    return (
        total_loss / num_splits,
        total_acc / num_splits,
        total_time / num_splits
    )

loss, acc, t = calculator(scores, times)


# ===== GUARDADO CSV =====
os.makedirs("resultados", exist_ok=True)

data = [{
    "Polinomio": "Lineal/Dense",
    "Grado": "N/A",
    "Mejor_N": "N/A",
    "Loss_Promedio": round(loss, 8),
    "Accuracy_Promedio": round(acc, 8),
    "Tiempo_Promedio(s)": round(t, 2)
}]

pd.DataFrame(data).to_csv("resultados/resultados_lineal.csv", index=False, sep=';')


# ===== GRÁFICAS =====
plot_cv_average_history(histories)

print("Proceso finalizado correctamente")
