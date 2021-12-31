from ..base.base_inverter import BaseInverter
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as kb

class SWEs2DModelInverter(BaseInverter):
    def __init__(self, modelWrapper, dataLoader, config):
        super(SWEs2DModelInverter, self).__init__(modelWrapper, dataLoader, config)

        #shhape of input (zb) and output (uvWSE) of the NN surrogate model
        self.input_data_shape = dataLoader.get_input_data_shape()
        self.output_data_shape = dataLoader.get_output_data_shape()

        #load target uvWSE
        self.uvWSE_target_np = dataLoader.get_uvWSE_inversion()
        self.uvWSE_target_np = self.uvWSE_target_np[np.newaxis, :, :, :]   #expand one more dimension (not necessary?)
        self.uvWSE_target = tf.Variable(self.uvWSE_target_np, trainable=False, name="uvWSE_target", dtype=np.float32)

        self.uvWSE_pred = None

        #the inversion variable, i.e., zb
        #self.zb_np = np.zeros(self.input_data_shape)
        self.zb_np = (np.random.random(self.input_data_shape)-0.5)*0.0

        #expand one dimension to zb_np, e.g, [64, 256, 1] to [1, 64, 256, 1]
        self.zb_np = self.zb_np[np.newaxis, :, :, :]

        self.zb = tf.Variable(self.zb_np, trainable=True, name="zb", dtype=np.float32)

        # Is the tape that computes the gradients!
        #self.trainable_variables = self.zb

        # Regularizer
        self.regularizer = tf.keras.regularizers.L2(self.config.inverter.L2_regularization_factor)

        # Optimizer
        self.optimizer = None
        if self.config.inverter.optimizer == "adam":
            self.optimizer = tf.optimizers.Adam(learning_rate=self.config.inverter.adam.learning_rate,
                                                epsilon=self.config.inverter.adam.epsilon)
        elif self.config.inverter.optimizer == "SGD":
            self.optimizer = tf.optimizers.SGD(learning_rate=self.config.inverter.learning_rate, momentum=0.9)
        else:
            raise Exception("Specified inversion optimizer not recognized.")

        # loss (error)
        self.loss = 0.0

        # record the loss history
        self.losses = []

    def invert(self):
        """
        Perform inversion to get the bed

        :return:
        """

        for i in range(self.config.inverter.nSteps):
            with tf.GradientTape() as tape:
                # use the surrogate NN model to make a prediction
                self.uvWSE_pred = self.modelWrapper.model(self.zb, training=False)

                # difference between target and predicted WSE values
                error = self.uvWSE_target - self.uvWSE_pred  # the error
                sqr_error = kb.square(error)  # square of the error
                self.loss = kb.sum(sqr_error)  # mean of the square of the error

                error_loss = self.loss

                # add regularization
                loss_regularization = self.regularizer(self.zb)

                self.loss += loss_regularization

                print("Iter. #, total loss, error loss, regularization: ",
                      i, self.loss.numpy(), error_loss.numpy(), loss_regularization.numpy())

                self.losses.append(self.loss.numpy())

            grads = tape.gradient(self.loss, [self.zb])

            self.optimizer.apply_gradients(zip(grads, [self.zb]))

        #save the loss history to file
        #np.savez("inversion_loss_history.npz", loss=self.losses)

    def get_zb(self):
        return self.zb.numpy()[0]

    def get_uvWSE_pred(self):
        return self.uvWSE_pred.numpy()[0]

    def get_uvWSE_target(self):
        return self.uvWSE_target_np

    def get_inversion_loss_history(self):
        return self.losses
