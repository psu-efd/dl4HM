from ..base.base_modelWrapper import BaseModelWrapper
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from tensorflow.keras import backend as kb
import tensorflow as tf

import  sys


class SWEs2DModel(BaseModelWrapper):
    """A NN model for 2D Shallow Water Equations solver's surrogate

    Model input variables: bed bathymetry zb
    Model output variables: u, v, WSE

    """

    def __init__(self, config, dataLoader):
        super(SWEs2DModel, self).__init__(config, dataLoader)

        self.build_model()

        #loss components (record each iteration, i.e., each batch). Total numbers = number of epoch X number of batches
        self.loss_value = []         #loss due to value error

        # loss components (record each epoch); will be set by a callback function in trainer
        self.loss_value_epoch = []         #loss due to value error

    def build_model(self):
        self.model = self.fully_connected_MLP_model()

        self.model.compile(
            loss=self.SWEs2DLossFunction(),
            optimizer=self.config.model.optimizer,
            run_eagerly=True,
        )

        # summarize the model
        self.model.summary()
        # plot_model(self.model, 'model.png', show_shapes=True)

    def fully_connected_MLP_model(self):
        """
        Fully connected multi-layer percentron model
        :return:
        """

        def fully_connected(input):
            # Arguments:
            # input -- input layer for the network, expected shape (?,nh,nw,1)
            # Returns -- predicted flow (?, nh, nw, 3)

            nh = kb.int_shape(input)[1]
            nw = kb.int_shape(input)[2]

            # define the hidden layer
            x = layers.Flatten()(input)
            x = layers.Dense(128, activation='relu')(x)

            # Define output layer and reshape it to nh x nw x 3.
            # (Note that the extra batch dimension is handled automatically by Keras)
            x = layers.Dense(nh * nw * 3, activation='relu')(x)
            output = layers.Reshape((nh, nw, 3))(x)

            return output

        # Define Inputs and Outputs
        input = tf.keras.Input(shape=tuple(self.dataLoader.input_data_shape))  #use self.dataLoader.get_input_data_shape()
        output = fully_connected(input)

        # Use Keras Functional API to Create our Model
        fc_model = tf.keras.Model(inputs=input, outputs=output)

        return fc_model

    def SWEs2DLossFunction(self):
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

            loss = tf.nn.l2_loss(y_true - y_pred)

            # Add a scalar to tensorboard
            tf.summary.scalar('loss', loss)

            self.loss_value.append(loss)

            return loss

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
