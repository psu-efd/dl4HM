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

        #the inversion variable, i.e., zb
        zb_np = np.zeros(self.input_data_length)

        self.zb = tf.Variable(zb_np, name='zb', trainable=True, dtype=tf.float32)

        # Is the tape that computes the gradients!
        self.trainable_variables = [self.zb]

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

        for i in range(100):
            train = self.optimizer.minimize(self.compute_loss, var_list=self.trainable_variables)



    def compute_loss(self):
        """
        Compute the loss
        :return:
        """

        #use the surrogate NN model to make a prediction
        WSE_pred = self.modelWrapper.model.predict(self.zb)

        # difference between target and predicted WSE values
        error = self.WSE_target - WSE_pred  # the error
        sqr_error = kb.square(error)  # square of the error
        self.loss = kb.mean(sqr_error)  # mean of the square of the error

        return self.loss

    def get_zb(self):
        return self.zb
