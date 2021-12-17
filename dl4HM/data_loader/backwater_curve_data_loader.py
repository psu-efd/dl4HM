from ..base.base_data_loader import BaseDataLoader
import numpy as np
import json


class BackwaterCurveDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(BackwaterCurveDataLoader, self).__init__(config)

        #load data
        self.x_bed_train, self.zb_beds_train, self.x_train, self.WSE_train, self.H_train, self.U_train, \
        self.x_bed_test, self.zb_beds_test, self.x_test, self.WSE_test, self.H_test, self.U_test \
            = self.load_data()

        #load optional inversion data
        self.WSE_inversion = None
        self.zb_beds_truth = None

        self.load_inversion_data()

        #scaling transformation
        self.scaled_data = self.config.dataLoader.scaled_data

        # load the min and max of all variables for scaling (if data are scaled)
        self.zb_beds_min = 0.0
        self.zb_beds_max = 0.0
        self.WSE_min = 0.0
        self.WSE_max = 0.0

        if self.scaled_data:
            with open(self.config.dataLoader.minMaxVars_file) as json_file:
                minMaxVars = json.load(json_file)

            self.zb_beds_min = minMaxVars['zb_beds_min']
            self.zb_beds_max = minMaxVars['zb_beds_max']
            self.WSE_min = minMaxVars['WSE_min']
            self.WSE_max = minMaxVars['WSE_max']

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

    def load_inversion_data(self):
        """Load inversion (if specified in the config file)

        :return:
        """

        if self.config.inverter.do_inversion:
            self.WSE_inversion = np.load(self.config.inverter.inversion_data)['WSE']
            self.zb_beds_truth = np.load(self.config.inverter.inversion_data)['zb_beds']
        else:
            self.WSE_inversion = None

    def forward_scale_x(self, x):
        """Forward scale x: here x denote input, i.e., the bed elevations zb_beds

        :param x:
        :return:
        """

        return self.scale_zb_beds_train.fit_transform(x)

    def backward_scale_x(self, x):
        """Backward scale x

        :param x:
        :return:
        """

        return self.scale_zb_beds_train.inverse_transform(x)

    def get_input_data_length(self):
        return self.input_data_length

    def get_output_data_length(self):
        return self.output_data_length

    def get_train_data(self):
        return self.zb_beds_train, self.WSE_train

    def get_test_data(self):
        return self.zb_beds_test, self.WSE_test

    def get_WSE_inversion(self):
        return self.WSE_inversion

    def get_zb_beds_truth(self):
        return self.zb_beds_truth

    def get_x_train(self):
        return self.x_train

    def get_x_bed_train(self):
        return self.x_bed_train