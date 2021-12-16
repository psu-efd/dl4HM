from ..base.base_modelWrapper import BaseModelWrapper
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

from tensorflow.keras import backend as kb
import tensorflow as tf

import  sys


class BackwaterCurveModel(BaseModelWrapper):
    """A NN model for backwater curve solver's surrogate

    Model input variables: bed profile zb
    Model output variables: WSE

    """

    def __init__(self, config, dataLoader):
        super(BackwaterCurveModel, self).__init__(config, dataLoader)

        self.build_model()

        #loss components (record each iteration, i.e., each batch). Total numbers = number of epoch X number of batches
        self.loss_value = []         #loss due to value error

        # loss components (record each epoch); will be set by a callback function in trainer
        self.loss_value_epoch = []         #loss due to value error

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=self.dataLoader.get_input_data_length(), activation='relu', \
                             kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.L1(0.0), \
                             activity_regularizer=tf.keras.regularizers.L2(0.0)))
        #self.model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(self.dataLoader.get_output_data_length(), activation='linear'))

        # summarize the model
        #plot_model(self.model, 'model.png', show_shapes=True)

        self.model.compile(
            loss=self.backwaterCurveLossFunction(),
            optimizer=self.config.model.optimizer,
            run_eagerly=True,
        )

    def backwaterCurveLossFunction(self):
        """Customized loss function for backwater curve


        :return:
        """

        #tf.print("\n x_for_gradient: ", type(x_for_loss_function), output_stream=sys.stdout)

        def loss(y_true, y_pred):
            """This is the loss function

            It seems that only this portion of the loss function is called during
            training. Anything outside is only called once during the "compile" stage.

            :param y_true:
            :param y_pred:
            :return:
            """

            # difference between true value and predicted value
            error = y_true - y_pred  # the error
            sqr_error = kb.square(error)  # square of the error
            mean_sqr_error = kb.mean(sqr_error)  # mean of the square of the error

            self.loss_value.append(mean_sqr_error)

            return mean_sqr_error

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
