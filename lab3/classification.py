import argparse
import sys

import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers, initializers
from matplotlib import pyplot as plt


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


def prepare_training_data():
    df = pd.read_csv(DATA_NAME, names=COLUMN_NAMES)
    df = df.sample(frac=1).reset_index(drop=True)

    # Dane X
    X = df.drop("Class", axis=1).values
    # Dane y - klasy wina
    y = df["Class"].values - 1

    # One-hot encoding dla klas wina
    y = tf.keras.utils.to_categorical(y, num_classes=3)

    # Standardyzacja danych
    global scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Podział danych trening/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def build_model():
    # Prosty model z jedną warstwą ukrytą
    model = Sequential(name="model_simple")
    model.add(layers.Input(shape=(13,), name="input"))
    model.add(layers.Dense(32, activation="relu", name="hidden"))
    model.add(layers.Dense(3, activation="softmax", name="output"))
    return model


def build_model_complex():
    # Eksperymentalny model z dwiema warstwami ukrytymi i dropoutem
    model = Sequential(name="model_complex")
    model.add(layers.Input(shape=(13,), name="input"))
    model.add(
        layers.Dense(
            64,
            activation="elu",
            kernel_initializer=initializers.HeUniform(),
            name="hidden1",
        )
    )
    model.add(layers.Dropout(0.3, name="dropout"))
    model.add(
        layers.Dense(
            32,
            activation="relu",
            kernel_initializer=initializers.HeNormal(),
            name="hidden2",
        )
    )
    model.add(layers.Dense(3, activation="softmax", name="output"))
    return model


def train_and_evaluate(
    model, X_train, X_test, y_train, y_test, epochs=50, lr=0.001, batch_size=16
):
    # Kompilacja modelu
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

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


def plot_history(history, model_name):

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{model_name}.png")
    plt.show()


def compare_models():

    X_train, X_test, y_train, y_test = prepare_training_data()

    simple_model = build_model()
    complex_model = build_model_complex()

    simple_model, simple_history, simple_loss, simple_accuracy = train_and_evaluate(
        simple_model, X_train, X_test, y_train, y_test
    )
    complex_model, complex_history, complex_loss, complex_accuracy = train_and_evaluate(
        complex_model, X_train, X_test, y_train, y_test
    )

    plot_history(simple_history, simple_model.name)
    plot_history(complex_history, complex_model.name)

    better_model = simple_model if simple_accuracy > complex_accuracy else complex_model
    print(f"Lepszy model: {better_model.name}")
    better_model.save("best_wine_model.keras")
    joblib.dump(scaler, "scaler_wine.pkl")


def predict_wine_class():

    parser = argparse.ArgumentParser(
        description="Przewiduje klasę wina na podstawie 13 cech."
    )
    parser.add_argument("--alcohol", type=float, required=True)
    parser.add_argument("--malic_acid", type=float, required=True)
    parser.add_argument("--ash", type=float, required=True)
    parser.add_argument("--alcalinity", type=float, required=True)
    parser.add_argument("--magnesium", type=float, required=True)
    parser.add_argument("--phenols", type=float, required=True)
    parser.add_argument("--flavanoids", type=float, required=True)
    parser.add_argument("--nonflavanoid", type=float, required=True)
    parser.add_argument("--proanthocyanins", type=float, required=True)
    parser.add_argument("--color", type=float, required=True)
    parser.add_argument("--hue", type=float, required=True)
    parser.add_argument("--od_ratio", type=float, required=True)
    parser.add_argument("--proline", type=float, required=True)

    args = parser.parse_args(sys.argv[1:])
    input_data = [
        [
            args.alcohol,
            args.malic_acid,
            args.ash,
            args.alcalinity,
            args.magnesium,
            args.phenols,
            args.flavanoids,
            args.nonflavanoid,
            args.proanthocyanins,
            args.color,
            args.hue,
            args.od_ratio,
            args.proline,
        ]
    ]

    model = load_model("best_wine_model.keras")
    scaler = joblib.load("scaler_wine.pkl")
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    print(f"Przewidywana klasa wina: {predicted_class + 1}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        compare_models()
    else:
        predict_wine_class()
