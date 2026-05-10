import time
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from metrics_utils import calculate_metrics, summarize_metrics


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
        prev_prev = tf.ones_like(inputs)
        prev = inputs

        output = (
            tf.matmul(prev_prev, self.w[0]) +
            tf.matmul(prev, self.w[1])
        )

        for n in range(2, self.degree + 1):
            current = 2.0 * inputs * prev - prev_prev
            output += tf.matmul(current, self.w[n])
            prev_prev = prev
            prev = current

        return output


class PolynomialLegendre(layers.Layer):
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
        prev_prev = tf.ones_like(x)
        prev = x

        features = [prev]
        for n in range(2, self.degree + 1):
            n = tf.cast(n, self.compute_dtype)
            current = ((2 * n - 1) * x * prev - (n - 1) * prev_prev) / n
            features.append(current)
            prev_prev = prev
            prev = current

        basis = tf.concat(features, axis=-1)
        output = tf.matmul(basis, self.kernel)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return output


class ShmaliyLayer(layers.Layer):
    def __init__(self, units, degree, N=100, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.degree = degree
        self.N = float(N)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.degree + 1, input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="shmaliy_weights"
        )

    def call(self, inputs):
        discrete_positions = (inputs + 1.0) * (self.N - 1.0) / 2.0
        prev_prev = tf.ones_like(discrete_positions)
        prev = 1.0 - (2.0 * discrete_positions) / (self.N - 1.0)

        output = (
            tf.matmul(prev_prev, self.w[0]) +
            tf.matmul(prev, self.w[1])
        )

        for k in range(1, self.degree):
            denom = (k + 1.0) * (self.N - 1.0 - k)
            term1 = ((2.0 * k + 1.0) * (self.N - 1.0 - 2.0 * discrete_positions) * prev) / denom
            term2 = (k * (self.N + k) * prev_prev) / denom
            current = term1 - term2

            output += tf.matmul(current, self.w[k + 1])
            prev_prev = prev
            prev = current

        return output


def create_early_stopping(patience=15):
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )


def build_lineal_model(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="LinealModel")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


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


def build_legendre_model(degree, input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    x = PolynomialLegendre(64, degree=degree)(inputs)
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


def build_shmaliy_model(degree, n_value, input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    x = ShmaliyLayer(64, degree=degree, N=n_value)(inputs)
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


def run_lineal_cv(X, y, n_folds, epochs, batch_size, patience):
    scores = []
    histories = []
    times = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.3,
            stratify=y_train,
            random_state=42
        )

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_sub = scaler.fit_transform(X_sub)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        num_classes = len(np.unique(y_train))
        tf.keras.backend.clear_session()
        model = build_lineal_model(X_sub.shape[1], num_classes)

        start = time.time()
        history = model.fit(
            X_sub,
            y_sub,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[create_early_stopping(patience)]
        )
        end = time.time()

        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        metrics = calculate_metrics(y_test, y_pred)
        scores.append(metrics)
        histories.append(history)
        times.append(end - start)

        del model
        K.clear_session()
        gc.collect()

    return summarize_metrics(scores), histories, float(np.mean(times))


def run_chebyshev_cv(X, y, n_folds, degrees, epochs, batch_size, patience):
    results = {}
    histories = {}
    times = {}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    for deg in degrees:
        histories[deg] = []
        times[deg] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.3,
            stratify=y_train,
            random_state=42
        )

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_sub = scaler.fit_transform(X_sub)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        num_classes = len(np.unique(y_train))

        for deg in degrees:
            tf.keras.backend.clear_session()
            model = build_chebyshev_model(deg, X_sub.shape[1], num_classes)

            start = time.time()
            history = model.fit(
                X_sub,
                y_sub,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[create_early_stopping(patience)]
            )
            end = time.time()

            y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
            metrics = calculate_metrics(y_test, y_pred)
            results.setdefault(deg, []).append(metrics)
            histories[deg].append(history)
            times[deg].append(end - start)

            del model
            K.clear_session()
            gc.collect()

    summary = {deg: summarize_metrics(results[deg]) for deg in degrees}
    avg_times = {deg: float(np.mean(times[deg])) for deg in degrees}
    return summary, histories, avg_times


def run_legendre_cv(X, y, n_folds, degrees, epochs, batch_size, patience):
    results = {}
    histories = {}
    times = {}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    for deg in degrees:
        histories[deg] = []
        times[deg] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.3,
            stratify=y_train,
            random_state=42
        )

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_sub = scaler.fit_transform(X_sub)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        num_classes = len(np.unique(y_train))

        for deg in degrees:
            tf.keras.backend.clear_session()
            model = build_legendre_model(deg, X_sub.shape[1], num_classes)

            start = time.time()
            history = model.fit(
                X_sub,
                y_sub,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[create_early_stopping(patience)]
            )
            end = time.time()

            y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
            metrics = calculate_metrics(y_test, y_pred)
            results.setdefault(deg, []).append(metrics)
            histories[deg].append(history)
            times[deg].append(end - start)

            del model
            K.clear_session()
            gc.collect()

    summary = {deg: summarize_metrics(results[deg]) for deg in degrees}
    avg_times = {deg: float(np.mean(times[deg])) for deg in degrees}
    return summary, histories, avg_times


def run_shmaliy_cv(X, y, n_folds, degrees, n_candidates, epochs, batch_size, patience):
    search_results = {d: {} for d in degrees}
    skf_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    for deg in degrees:
        for n_val in n_candidates:
            accs = []

            for train_idx, test_idx in skf_search.split(X, y):
                tf.keras.backend.clear_session()
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = MinMaxScaler(feature_range=(-1, 1))
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                num_classes = len(np.unique(y_train))
                model = build_shmaliy_model(deg, n_val, X.shape[1], num_classes)

                model.fit(
                    X_train,
                    y_train,
                    epochs=30,
                    batch_size=64,
                    verbose=0,
                    validation_split=0.1
                )

                y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
                metrics = calculate_metrics(y_test, y_pred)
                accs.append(metrics["accuracy"])

                del model
                K.clear_session()
                gc.collect()

            search_results[deg][n_val] = float(np.mean(accs))

    results = {}
    histories = {}
    times = {}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    for deg in degrees:
        histories[deg] = []
        times[deg] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.3,
            stratify=y_train,
            random_state=42
        )

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_sub = scaler.fit_transform(X_sub)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        num_classes = len(np.unique(y_train))

        for deg in degrees:
            tf.keras.backend.clear_session()
            best_n = max(search_results[deg], key=search_results[deg].get)
            model = build_shmaliy_model(deg, best_n, X_sub.shape[1], num_classes)

            start = time.time()
            history = model.fit(
                X_sub,
                y_sub,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[create_early_stopping(patience)]
            )
            end = time.time()

            y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
            metrics = calculate_metrics(y_test, y_pred)
            results.setdefault(deg, []).append(metrics)
            histories[deg].append(history)
            times[deg].append(end - start)

            del model
            K.clear_session()
            gc.collect()

    summary = {deg: summarize_metrics(results[deg]) for deg in degrees}
    avg_times = {deg: float(np.mean(times[deg])) for deg in degrees}
    return summary, histories, avg_times, search_results
