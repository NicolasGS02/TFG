#Primero de todo importamos las librerías necesarias para el proyecto

#Librerias para la creacion de los modelos
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler

#Librerias para la carga de los datos
from ucimlrepo import fetch_ucirepo 

#Libreria para dibujar graficos
import matplotlib.pyplot as plt

#Crear CSV
import pandas as pd
import os

#creamos el modelo de la red neuronal
def create_model(input_dim):
    x1 = keras.Input(shape=(input_dim,))
    
    x2 = layers.Dense(64, activation='relu')(x1)

    x3 = layers.Dense(64, activation='relu')(x2)
    
    output = layers.Dense(2, activation='softmax')(x3)
    
    model = keras.Model(
        inputs=x1,
        outputs=output,
        name='LinealModel'
    )
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ===== FUNCIÓN DE PLOTEO =====
def plot_training_history_leg(history_leg):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_leg.history['loss'], label='Pérdida entrenamiento')
    plt.plot(history_leg.history['val_loss'], label='Pérdida validación')
    plt.title('Pérdida')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_leg.history['accuracy'], label='Precisión entrenamiento')
    plt.plot(history_leg.history['val_accuracy'], label='Precisión validación')
    plt.title('Precisión')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


#Importamos el dataSet
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np


magic_gamma_telescope = fetch_ucirepo(id=159) 
  
X = magic_gamma_telescope.data.features 
y = magic_gamma_telescope.data.targets 

X = X.to_numpy()
y = y.to_numpy()


acurracy_scores = []

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in tqdm(skf.split(X, y), total=10):


    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #Normalizamos los datos para que el entrenamiento sea más eficiente entre -1 y 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train.shape[1]


    #Cambiamos las etiquetas a 0 y 1
    y_train = (y_train == 'g').astype(int)
    y_test = (y_test == 'g').astype(int)

    model = create_model(X_train_scaled.shape[1])
    trained_model = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=120, batch_size=32, verbose=0)
    eval_result = model.evaluate(X_test_scaled, y_test, verbose=0)
    _, accuracy = eval_result
    acurracy_scores.append(accuracy)



# 1. Definimos la función con la estructura que pediste
def save_results_to_csv_Lineal(avg_loss, avg_acc, filename="temp_res_lineal.csv"):
    # Definir la ruta de la carpeta (un nivel arriba '..', carpeta 'resultados')
    # O simplemente "resultados" si prefieres que esté en la misma carpeta
    carpeta_destino = os.path.join("..", "resultados")
    os.makedirs(carpeta_destino, exist_ok=True)
    ruta_completa = os.path.join(carpeta_destino, filename)
    
    # Creamos la lista con el formato compatible
    data = [{
        "Polinomio": "Lineal/Dense",
        "Grado": "N/A",
        "Mejor_N": "N/A",
        "Loss_Promedio": round(avg_loss, 6),
        "Accuracy_Promedio": round(avg_acc, 6)
    }]
    
    df_resultados = pd.DataFrame(data)
    df_resultados.to_csv(ruta_completa, index=False, sep=';')

save_results_to_csv_Lineal(np.mean(acurracy_scores), np.mean(acurracy_scores))