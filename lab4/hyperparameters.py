#!/usr/bin/python

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

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


df = pd.read_csv(DATA_NAME, names=COLUMN_NAMES)
df = df.sample(frac=1).reset_index(drop=True)

X = df.drop("Class", axis=1).values
y = df["Class"].values - 1
y = tf.keras.utils.to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X)


def build_baseline():
    model = Sequential(name="baseline_model")
    model.add(normalizer)
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_tuner_model(hp):
    model = Sequential(name="tuner_model")
    model.add(normalizer)
    model.add(layers.Dense(hp.Choice("units", [8, 16, 32, 64]), activation="relu"))
    model.add(layers.Dropout(hp.Float("dropout", 0.0, 0.6, step=0.1)))
    model.add(layers.Dense(3, activation="softmax"))

    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(model, name, save_path):
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        verbose=0,
        validation_data=(X_test, y_test),
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model: {name}")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Acc: {acc:.4f}")

    model.summary()

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print("Macierz pomylek:")
    print(cm)

    model.save(save_path)

    return history, acc


def main():
    baseline = build_baseline()
    baseline_history, baseline_acc = train_and_evaluate(
        baseline, "Baseline Model", "baseline_model.keras"
    )

    tuner = kt.RandomSearch(
        build_tuner_model,
        objective="val_loss",
        max_trials=10,
        project_name="wine_hyperparameters",
    )

    tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    best_hp = tuner.get_best_hyperparameters()[0]
    print("\nNajlepsze hiperparametry:")
    print("Ilość neuronów w warstwie ukrytej:", best_hp.get("units"))
    print("Poziom dropoutu:", best_hp.get("dropout"))
    print("Learning rate:", best_hp.get("lr"))

    best_model = tuner.hypermodel.build(best_hp)

    tuner_history, tuner_acc = train_and_evaluate(
        best_model, "Tuner Model", "tuner_model.keras"
    )

    print(f"Baseline acc: {baseline_acc:.4f}")
    print(f"Tuner acc:    {tuner_acc:.4f}")

if __name__ == "__main__":
    main()
