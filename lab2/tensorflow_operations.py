import argparse
import sys

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--degree", type=int, default=30)
args = parser.parse_args()

# Zadanie 1/2.
@tf.function
def rotate(points, theta):
    rotation_matrix = tf.stack(
        [tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)]
    )
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    print(rotation_matrix)
    return tf.matmul(rotation_matrix, points)

print(rotate(tf.constant([2.0, 0.0], shape=(2,1)), np.deg2rad(args.degree, dtype=np.float32)))

# Zadanie 3

def main() -> int:
    return 0

if __name__ == "__main__":
    sys.exit(main())
