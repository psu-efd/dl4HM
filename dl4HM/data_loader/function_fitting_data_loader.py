from ..base.base_data_loader import BaseDataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


class FunctionFittingDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(FunctionFittingDataLoader, self).__init__(config)

        #lower and upper bounds of x
        self.x_min = config.case.x_min
        self.x_max = config.case.x_max
        self.num_points = config.case.num_points

        #scaling transformation
        self.scale_x = MinMaxScaler()
        self.scale_y = MinMaxScaler()

        self.x_train, self.y_train, self.x_test, self.y_test = self.generate_data()

        #for the calculation of loss function
        self.x_for_loss_function = self.generate_x_for_loss_function()

    def generate_data(self):
        """Generate training and test data

        This is a simple example to fit a function.

        Reference: https://machinelearningmastery.com/neural-networks-are-function-approximators/

        :return:
        """

        # define the function here
        x = np.linspace(self.x_min, self.x_max, self.num_points)
        y = x ** 2.0

        print(x.min(), x.max(), y.min(), y.max())

        # reshape arrays into into rows and cols
        x = x.reshape((len(x), 1))
        y = y.reshape((len(y), 1))

        # separately scale the input and output variables
        x_train = self.forward_scale_x(x)
        y_train = self.forward_scale_y(y)
        print(x_train.min(), x_train.max(), y_train.min(), y_train.max())

        return x_train, y_train, [], []
        #return x_train, y_train, x_train, y_train

    def forward_scale_x(self, x):
        """Forward scale x

        :param x:
        :return:
        """

        return self.scale_x.fit_transform(x)

    def forward_scale_y(self, y):
        """Forward scale y

        :param y:
        :return:
        """

        return self.scale_y.fit_transform(y)


    def backward_scale_x(self, x):
        """Backward scale x

        :param x:
        :return:
        """

        return self.scale_x.inverse_transform(x)

    def backward_scale_y(self, y):
        """Backward scale y

        :param y:
        :return:
        """

        return self.scale_y.inverse_transform(y)

    def generate_x_for_loss_function(self):
        """

        :return:

        """

        x_func = np.linspace(self.x_min, self.x_max, self.num_points)

        # reshape arrays into into rows and cols
        x_func = x_func.reshape((len(x_func), 1))

        # separately scale the input and output variables
        x_func = self.forward_scale_x(x_func)

        return tf.Variable(np.expand_dims(x_func, axis=1), dtype=np.float32)


    def get_train_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_x_for_loss_function(self):
        return self.x_for_loss_function

