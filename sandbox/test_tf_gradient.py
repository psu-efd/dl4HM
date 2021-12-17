"""
A simple test for using TF to calculate the gradient of a NN's output w.r.t its input. Such gradient can be used by an
optimizer to do minimization.
"""

import tensorflow as tf

x = tf.Variable(2, name='x', trainable=True, dtype=tf.float32)

# Is the tape that computes the gradients!
trainable_variables = [x]

# To use minimize you have to define your loss computation as a funcction
class Model():
    def __init__(self):
        self.y = 0

    def compute_loss(self):
        self.y = tf.math.square(x)
        return self.y

opt = tf.optimizers.Adam(learning_rate=0.01)

model = Model()

for i in range(1000):
    train = opt.minimize(model.compute_loss, var_list=trainable_variables)

print("x:", x)
print("y:", model.y)