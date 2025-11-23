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
from sklearn.metrics import confusion_matrix
import numpy as np

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

    X = df.drop("Class", axis=1).values
    y = df["Class"].values - 1
    y = tf.keras.utils.to_categorical(y, num_classes=3)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    global data_normalized_layer
    data_normalized_layer = tf.keras.layers.Normalization()
    data_normalized_layer.adapt(X)

    return X_train, X_test, y_train, y_test


def build_baseline_model():
    model = Sequential(name="model_simple")
    model.add(data_normalized_layer)
    model.add(layers.Dense(32, activation="relu", name="hidden"))
    model.add(layers.Dense(3, activation="softmax", name="output"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_tuner_model(hp):
    model = Sequential(name="model_tuner")

    model.add(data_normalized_layer)
    model.add(layers.Dense(hp.Choice("units", [8, 16, 32, 64]), activation="relu"))
    model.add(layers.Dropout(hp.Float("dropout", min_value=0.0, max_value=0.6, step=0.1)))
    model.add(layers.Dense(3, activation="softmax"))

    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def run_tuner(X_train, X_test, y_train, y_test):
    tuner = kt.RandomSearch(
        build_tuner_model,
        objective="val_loss",
        max_trials=10,
        project_name="wine_hyperparameters",
    )
    tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    best_hps = tuner.get_best_hyperparameters()[0]

    print(f"Najlepsza ilość neuronów w warstwie ukrytej: {best_hps.get('units')}")
    print(f"Najlepszy learning rate: {best_hps.get('lr')}")
    print(f"Najlepszy dropout: {best_hps.get('dropout')}")

    best_model = tuner.hypermodel.build(best_hps)
    train_and_evaluate(best_model, X_train, X_test, y_train, y_test)
    best_model.save("tuner_model.keras")

def run_model(model):
    X_train, X_test, y_train, y_test = prepare_data()

    if model == "simple":
        model = build_baseline_model()
        train_and_evaluate(model, X_train, X_test, y_train, y_test)
    else:
        run_tuner(X_train, X_test, y_train, y_test)


def train_and_evaluate(
        model, X_train, X_test, y_train, y_test, epochs=50, batch_size=16
):
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_data=(X_test, y_test),
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(
        f"Model - {model.name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )
    model.summary()
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print(f"Macierz pomylek {cm}")

    return model, history, test_loss, test_accuracy



def main():
    parser = argparse.ArgumentParser(
        description="Wybór modelu (simple, model z tunerem)"
    )
    parser.add_argument("--model", type=str, required=True, default="simple", choices=["simple", "tuner"])
    args = parser.parse_args(sys.argv[1:])
    run_model(args.model)


if __name__ == "__main__":
    main()
