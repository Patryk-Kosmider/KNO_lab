import argparse
import sys

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--degree", type=int, default=30)
parser.add_argument("-n", type=int)
parser.add_argument("-A", type=int, nargs="+", required=True)
parser.add_argument("-b", type=int, nargs="+", required=True)
args = parser.parse_args()

# Zadanie 1/2.
@tf.function
def rotate(points, theta):
    rotation_matrix = tf.stack(
        [tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)]
    )
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    return tf.matmul(rotation_matrix, points)

#print(rotate(tf.constant([[2.0, 0.0]], shape=(2,1)), np.deg2rad(args.degree, dtype=np.float32)))


def solve_linear_system(A, b):
    det = tf.linalg.det(A)
    if det == 0:
        raise ValueError("Układ nie ma rozwiązań, lub ma nieskończenie wiele / wyznacznik równy 0")
    x = tf.linalg.solve(A, b)
    return x
A = tf.constant(args.A,shape=(args.n, args.n), dtype=tf.float32)
b = tf.constant(args.b,shape=(args.n,1), dtype=tf.float32)
solution = solve_linear_system(A, b)
solution = solution.numpy().flatten()
print(solution)


def main() -> int:
    return 0

if __name__ == "__main__":
    sys.exit(main())
