#!/usr/bin/python

import argparse
import sys

import pandas as pd
import tensorflow as tf
import joblib
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers, initializers
from matplotlib import pyplot as plt

"""
Model - model_simple - Test Loss: 0.0545, Test Accuracy: 0.9722
Model - model_complex - Test Loss: 0.0346, Test Accuracy: 0.9722
"""

DATA_NAME = "wine.data"
COLUMN_NAMES = [
    "Class",
    "Alcohol",
    "Malicacid",
    "Ash",
    "Alcalinity_of_ash",
    "Magnesium",
    "Total_phenols",
    "Flavanoids",
    "Nonflavanoid_phenols",
    "Proanthocyanins",
    "Color_intensity",
    "Hue",
    "OD280/OD315_of_diluted_wines",
    "Proline",
]


def prepare_data():
    df = pd.read_csv(DATA_NAME, names=COLUMN_NAMES)
    df = df.sample(frac=1).reset_index(drop=True)

    # Dane X
    X = df.drop("Class", axis=1).values
    # Dane y - klasy wina
    y = df["Class"].values - 1

    # One-hot encoding dla klas wina
    y = tf.keras.utils.to_categorical(y, num_classes=3)

    # Podział danych trening/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    global data_normalized_layer
    data_normalized_layer = tf.keras.layers.Normalization()
    data_normalized_layer.adapt(X)


    return X_train, X_test, y_train, y_test, data_normalized_layer

def build_model():
    model = Sequential(name="model_simple")
    model.add(data_normalized_layer)
    model.add(layers.Dense(32, activation="relu", name="hidden"))
    model.add(layers.Dense(3, activation="softmax", name="output"))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_tuner_model(hp):
    #TODO: optymalizacja units, dropouts, lr
    model = Sequential(name="model_tuner")
    model.add(data_normalized_layer)
    model.add(layers.Dense(hp.Choice('units', [8,16,32,64]), activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.2))

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling='log')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_and_evaluate(
    model, X_train, X_test, y_train, y_test, epochs=50, lr=0.001, batch_size=16
):
    if model == "tuner":
        tuner = kt.RandomSearch(build_tuner_model, objective="val_loss", max_trials=5)
        tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
        best_hps = tuner.get_best_hyperparameters()[0]
        print(best_hps)
        print(f"""
             Najlepsza ilość neuronów w warstwie ukrytej: {best_hps.get('units')}.\t
             Najlepszy learning rate: {best_hps.get('lr')}
            """
        )
    else:
        # Trenowanie modelu
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(X_test, y_test),
        )
    # Ewaluacja modelu
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        print(
            f"Model - {model.name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

        return model, history, test_loss, test_accuracy




def run_model(model):
    # Wybor modelu do uruchomienia
    X_train, X_test, y_train, y_test, data_normalized_layer = prepare_data()
    
    if model == "simple":
        model = build_model()
    
    train_and_evaluate(model, X_train, X_test, y_train, y_test)



def main():

    parser = argparse.ArgumentParser(
        description="Wybór modelu (simple, model z tunerem)"
    )

    parser.add_argument("--model", type=str, required=True, default="simple")
    args = parser.parse_args(sys.argv[1:])

    run_model(args.model)

if __name__ == "__main__":
    main()

 