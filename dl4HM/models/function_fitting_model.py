from ..base.base_modelWrapper import BaseModelWrapper
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

from tensorflow.keras import backend as kb
import tensorflow as tf

import  sys


class FunctionFittingModel(BaseModelWrapper):
    """A NN model for fitting a fuction

    Model input variables: (x, y) coordinates
    Model output variables: (psi, p) where psi is the stream function and p is pressure

    """

    def __init__(self, config, dataLoader):
        super(FunctionFittingModel, self).__init__(config, dataLoader)

        self.build_model(dataLoader.get_x_for_loss_function())

        # record the current (x, dydx). Can be used to make plot at the end.
        self.x_dydx = None

        #loss components (record each iteration, i.e., each batch). Total numbers = number of epoch X number of batches
        self.loss_value = []         #loss due to value error
        self.loss_derivative = []    #loss due to derivative error

        # loss components (record each epoch); will be set by a callback function in trainer
        self.loss_value_epoch = []         #loss due to value error
        self.loss_derivative_epoch = []    #loss due to derivative error


    def build_model(self, x_for_loss_function):
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=1, activation='relu', kernel_initializer='he_uniform'))  #he_uniform
        self.model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(1, activation='linear'))

        # summarize the model
        plot_model(self.model, 'model.png', show_shapes=True)

        self.model.compile(
            #loss=self.functionFittingLossFunction(self.model.inputs), #this does not work (https://github.com/tensorflow/tensorflow/issues/36596)
            loss=self.functionFittingLossFunction(x_for_loss_function),
            optimizer=self.config.model.optimizer,
            run_eagerly=True,
        )

    def functionFittingLossFunction(self, x_for_loss_function):
        """Customized loss function for function fitting

        Reference: https://github.com/keras-team/keras/issues/2121


        :return:
        """

        tf.print("\n x_for_gradient: ", type(x_for_loss_function), output_stream=sys.stdout)

        def loss(y_true, y_pred):
            """This is the loss function

            It seems that only this portion of the loss function is called during
            training. Anything outside is only called once during the "compile" stage.

            :param y_true:
            :param y_pred:
            :return:
            """

            #tf.print("\n x_for_gradient: ", type(x_for_gradient), output_stream=sys.stdout)
            #tf.print("\n x_for_gradient[0]: ", type(x_for_gradient[0]), output_stream=sys.stdout)

            # gradient for derivatives of y wrt x
            with tf.GradientTape() as tape:
                tape.watch(x_for_loss_function)

                u = self.model(x_for_loss_function)

            dydx = tape.gradient(u, x_for_loss_function)

            #print(tf.executing_eagerly())

            self.x_dydx = tf.concat([x_for_loss_function, dydx], 1).numpy()

            #tf.print("\n dydx: ", dydx, output_stream=sys.stdout)
            #tf.print("\n x_dydx: \n", self.x_dydx, output_stream=sys.stdout, summarize=-1)

            # difference between true value and predicted value
            error = y_true - y_pred  # the error
            sqr_error = kb.square(error)  # square of the error
            mean_sqr_error = kb.mean(sqr_error)  # mean of the square of the error

            # derivative error
            derivative_error = 0*(dydx - 2*x_for_loss_function)   # the error
            sqr_derivative_error = kb.square(derivative_error)  # square of the error
            mean_sqr_error_derivative = kb.mean(sqr_derivative_error)  # mean of the square of the error

            self.loss_value.append(mean_sqr_error)
            self.loss_derivative.append(mean_sqr_error_derivative)

            return (mean_sqr_error + mean_sqr_error_derivative)

        return loss

    def test_model(self, test_data, verbose=0):
        """Test the model with given data

        :param test_data:
        :param verbose:
        :return:
        """

        if self.model is None:
            raise Exception("You have to build the model first.")

        score = self.model.evaluate(test_data[0], test_data[1], verbose=0)

        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def predict(self, input_data, verbose=0):
        """Use the trained model to make prediction

        :param input_data:
        :param verbose:
        :return:
        """

        if self.model is None:
            raise Exception("You have to build and train the model first.")

        return self.model.predict(input_data)
