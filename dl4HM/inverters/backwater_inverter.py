from ..base.base_inverter import BaseInverter
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as kb

class BackwaterCurveModelInverter(BaseInverter):
    def __init__(self, modelWrapper, dataLoader, config):
        super(BackwaterCurveModelInverter, self).__init__(modelWrapper, dataLoader, config)

        #length of input (zb) and output (WSE) of the NN surrogate model
        self.input_data_length = dataLoader.get_input_data_length()
        self.output_data_length = dataLoader.get_output_data_length

        #load target WSE data
        self.WSE_target = dataLoader.get_WSE_inversion()

        self.WSE_pred = None

        #the inversion variable, i.e., zb
        #self.zb_np = np.zeros(self.input_data_length)
        self.zb_np = np.random.random(self.input_data_length)*0.01

        self.zb = tf.Variable(np.expand_dims(self.zb_np, axis=0), dtype=np.float32)

        # Is the tape that computes the gradients!
        self.trainable_variables = self.zb

        # Optimizer
        self.optimizer = None
        if self.config.inverter.optimizer == "adam":
            self.optimizer = tf.optimizers.Adam(learning_rate=self.config.inverter.learning_rate)
        else:
            raise Exception("Specified inversion optimizer not recognized.")

        # loss (error)
        self.loss = 0.0

    def invert(self):
        """
        Perform inversion to get the bed profile

        :return:
        """

        for i in range(3000):
            with tf.GradientTape() as tape:
                # use the surrogate NN model to make a prediction
                self.WSE_pred = self.modelWrapper.model(self.zb)

                # difference between target and predicted WSE values
                error = self.WSE_target - self.WSE_pred  # the error
                sqr_error = kb.square(error)  # square of the error
                self.loss = kb.mean(sqr_error)  # mean of the square of the error

                print("Iteration # and loss: ", i, self.loss.numpy())

            grads = tape.gradient(self.loss, [self.zb])

            self.optimizer.apply_gradients(zip(grads, [self.zb]))

    def get_zb(self):
        return self.zb.numpy()[0]

    def get_WSE_pred(self):
        return self.WSE_pred.numpy()[0]

    def get_WSE_target(self):
        return self.WSE_target
