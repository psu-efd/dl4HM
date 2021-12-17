import tensorflow as tf
import numpy as np

@tf.function
def f(A, Y, X):
    AX = tf.linalg.matvec(A, X)
    norm = tf.norm(Y - AX)
    return norm

N = 2
A = tf.constant(np.array([[1., 2.], [3., 4.]]))
Y = tf.constant(np.array([3., 7.]))
X = tf.Variable(np.zeros(N))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for iteration in range(0, 300):
    with tf.GradientTape() as tape:
        loss = f(A, Y, X)
        print(loss)

    grads = tape.gradient(loss, [X])
    optimizer.apply_gradients(zip(grads, [X]))

print("A = ", A)
print("Y = ", Y)
print("X = ", X)