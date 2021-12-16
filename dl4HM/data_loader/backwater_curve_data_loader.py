from ..base.base_data_loader import BaseDataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


class BackwaterCurveDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(BackwaterCurveDataLoader, self).__init__(config)

        #scaling transformation
        self.scale_x = MinMaxScaler()
        self.scale_y = MinMaxScaler()

        #load data
        self.x_bed_train, self.zb_beds_train, self.x_train, self.WSE_train, self.H_train, self.U_train, \
        self.x_bed_test, self.zb_beds_test, self.x_test, self.WSE_test, self.H_test, self.U_test \
            = self.load_data()

        #sizes of input and output
        #input: zb_bed
        #output: WSE
        self.input_data_length = self.zb_beds_train.shape[1]
        self.output_data_length = self.WSE_train.shape[1]

    def load_data(self):
        """Load training and test data

        :return:
        """

        training_data = np.load(self.config.dataLoader.training_data)
        testing_data = np.load(self.config.dataLoader.testing_data)

        x_bed_train = training_data['x_bed']
        zb_beds_train = training_data['zb_beds']
        x_train = training_data['x']
        WSE_train = training_data['WSE']
        H_train = training_data['H']
        U_train = training_data['U']

        x_bed_test = testing_data['x_bed']
        zb_beds_test = testing_data['zb_beds']
        x_test = testing_data['x']
        WSE_test = testing_data['WSE']
        H_test = testing_data['H']
        U_test = testing_data['U']

        # separately scale the input and output variables
        # not implemented yet


        return x_bed_train, zb_beds_train, x_train, WSE_train, H_train, U_train, x_bed_test, zb_beds_test, \
                x_test, WSE_test, H_test, U_test

    def forward_scale_x(self, x):
        """Forward scale x

        :param x:
        :return:
        """

        return self.scale_x.fit_transform(x)

    def backward_scale_x(self, x):
        """Backward scale x

        :param x:
        :return:
        """

        return self.scale_x.inverse_transform(x)

    def get_input_data_length(self):
        return self.input_data_length

    def get_output_data_length(self):
        return self.output_data_length

    def get_train_data(self):
        return self.zb_beds_train, self.WSE_train

    def get_test_data(self):
        return self.zb_beds_test, self.WSE_test

    def get_x_train(self):
        return self.x_train

    def get_x_bed_train(self):
        return self.x_bed_train