import sys
import os
import keras.models
import numpy as np
import tensorflow as tf
import argparse
from keras.preprocessing import image

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
args = parser.parse_args()

# wymiary zdjecia
IMG_WIDTH, IMG_HEIGHT = 28, 28


def load_image(img_path):
    img = image.load_img(
        img_path, color_mode="grayscale", target_size=(IMG_WIDTH, IMG_HEIGHT)
    )
    img_tensor = image.img_to_array(img)
    img_tensor = np.array([img_tensor])
    return img_tensor


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if os.path.exists("./model.keras"):
    model = keras.models.load_model("model.keras")
else:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=5)  # użyj verbose=0 jeśli jest problem z konsolą
    model.evaluate(x_test, y_test)
    model.save("model.keras")

img = load_image(args.image)
pred = model.predict(img)

print(pred[0])


def main() -> int:
    return 0


if __name__ == "__main__":
    sys.exit(main())
