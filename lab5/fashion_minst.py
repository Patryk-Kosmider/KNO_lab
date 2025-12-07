import argparse

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def load_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    return train_images, train_labels, test_images, test_labels


def build_cnn_model(hp=None):

    filters_1 = hp.Int("filters_1", 32, 64, step=32) if hp else 32
    filters_2 = hp.Int("filters_2", 64, 128, step=32) if hp else 64
    dense = hp.Choice("dense", [32, 64, 128]) if hp else 64
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log") if hp else 0.001

    model = Sequential(name="model_cnn")
    model.add(
        layers.Conv2D(filters_1, (3, 3), activation="relu", input_shape=(28, 28, 1))
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters_2, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_dense_model(hp=None):
    dense_units1 = hp.Choice("dense", [32, 64, 128]) if hp else 128
    dense_units2 = hp.Choice("dense", [32, 64, 128]) if hp else 32
    dropout_rate = hp.Float("dropout", 0.0, 0.5, step=0.1) if hp else 0.2
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log") if hp else 0.001

    model = Sequential(name="dense_model")
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(dense_units1, activation="relu"))

    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(dense_units2, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_and_evaluate(model, data, save_path, epochs=5, batch_size=32):
    train_images, train_labels, test_images, test_labels = data

    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(test_images, test_labels),
    )

    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Model: {model.name}")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Acc: {acc:.4f}")

    model.summary()

    y_pred = np.argmax(model.predict(test_images), axis=1)
    cm = confusion_matrix(test_labels, y_pred)
    print("Macierz pomylek:")
    print(cm)

    model.save(save_path)

    with open(f"{model.name}_metrics.txt", "w") as f:
        f.write(f"Metryki modelu: {model.name}\n")
        f.write(f"Test Loss:     {loss:.4f}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write("Confusion Matrix:\n\n")

        f.write(str(cm))

    return history, acc


def run_tuner(model, data, project_name):
    train_images, train_labels, test_images, test_labels = data

    tuner = kt.RandomSearch(
        model,
        objective="val_loss",
        max_trials=5,
        project_name=project_name,
    )

    tuner.search(
        train_images, train_labels, epochs=5, validation_data=(test_images, test_labels)
    )
    best_hp = tuner.get_best_hyperparameters()[0]

    print(f"\nNajlepsze hiperparametry dla {project_name}:")
    for param, value in best_hp.values.items():
        print(f"{param}: {value}")

    best_model = tuner.hypermodel.build(best_hp)

    tuner_history, tuner_acc = train_and_evaluate(
        best_model,
        data,
        f"{project_name}_tuner.keras",
    )
    print(f"Tuner acc:    {tuner_acc:.4f}")


def main():
    data = load_data()

    parser = argparse.ArgumentParser()
    parser.add_argument("--tuner", type=bool, default=False)

    args = parser.parse_args()

    if args.tuner:
        run_tuner(build_dense_model, data, "dense_model_tuner")
        run_tuner(build_cnn_model, data, "cnn_model_tuner")
    else:
        fully_connected_model = build_dense_model()
        train_and_evaluate(
            fully_connected_model, data, "dense_model.keras", epochs=5, batch_size=32
        )

        cnn_model = build_cnn_model()
        train_and_evaluate(cnn_model, data, "cnn_model.keras", epochs=5, batch_size=32)


if __name__ == "__main__":
    main()
