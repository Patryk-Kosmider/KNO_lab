import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
])

def normalize(images):
    images = tf.cast(images, tf.float32) / 255.0
    return images, images

def load_data():
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "images",
        labels = None,
        image_size = (128, 128),
        batch_size = 8,
        shuffle = True,
        validation_split = 0.2,
        subset = "training",
        seed = 123,
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "images",
        labels = None,
        image_size = (128, 128),
        batch_size = 8,
        shuffle = True,
        validation_split = 0.2,
        subset = "validation",
        seed = 123,
    )

    train_dataset = train_dataset.map(normalize)
    validation_dataset = validation_dataset.map(normalize)

    return train_dataset, validation_dataset

def build_encoder_model(latent_dim=2, use_augmentation=False):
    model = Sequential(name="encoder")
    model.add(layers.Input(shape=(128, 128, 3)))

    if use_augmentation:
        model.add(data_augmentation)

    model.add(layers.Conv2D(64, (3, 3), activation="elu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Conv2D(128, (3, 3), activation="elu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Conv2D(256, (3, 3), activation="elu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(latent_dim, activation='linear'))
    return model

def build_decoder_model(latent_dim=2):
    model = Sequential(name="decoder")
    model.add(layers.Input(shape=(latent_dim,)))
    model.add(layers.Dense(16 * 16 * 256, activation="elu"))
    model.add(layers.Reshape((16, 16, 256)))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3),activation="elu", padding="same"))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="elu", padding="same"))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation="elu", padding="same"))
    model.add(layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same"))
    return model

def build_autoencoder_model(latent_dim=2, use_augmentation=False):
    encoder = build_encoder_model(latent_dim, use_augmentation)
    decoder = build_decoder_model(latent_dim)

    autoencoder = Sequential([encoder, decoder], name="autoencoder")
    autoencoder.compile(optimizer="adam", loss='mse')
    return autoencoder, encoder

def train_and_evaluate(model, train_data, val_data, epochs=50):
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
    )

    model.save("autoencoder.keras")
    return history

def show_reconstructions(model, dataset, n=5):
    for batch_x, _ in dataset.take(1):
        reconstructed = model.predict(batch_x)
        plt.figure(figsize=(10, 4))
        for i in range(min(n, batch_x.shape[0])):
            plt.subplot(2, n, i + 1)
            plt.imshow(batch_x[i])
            plt.axis('off')
            plt.title("Oryginał")

            plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed[i])
            plt.axis('off')
            plt.title("Wynik")
        plt.show()
        plt.savefig("rekonstrukcja.png")
        break

def main():
    parser = argparse.ArgumentParser(description="Train an autoencoder on images.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--use_augmentation", action="store_true")
    args = parser.parse_args()

    train_data, val_data = load_data()
    model, encoder = build_autoencoder_model(latent_dim=args.latent_dim, use_augmentation=args.use_augmentation)
    history = train_and_evaluate(model, train_data, val_data, epochs=args.epochs)

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig("loss.png")

    show_reconstructions(model, val_data)

if __name__ == "__main__":
    main()