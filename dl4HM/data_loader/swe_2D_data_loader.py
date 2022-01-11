from ..base.base_data_loader import BaseDataLoader
import numpy as np
import json

import tensorflow as tf


class SWEs2DDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SWEs2DDataLoader, self).__init__(config)

        #the total number of records in training, validation, and test data sets
        self.nTraining_data = -1
        self.nValidation_data = -1
        self.nTest_data = -1

        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        #calcualte how many batches in training, validation, and test
        self.nTraining_batches = -1
        self.nValidation_batches = -1
        self.nTest_batches = -1

        #sizes of input and output
        #input: zb_bed
        #output: (u, v, WSE)
        self.input_data_shape = None
        self.output_data_shape = None

        #inversion data (uvWSE): given this uvWSE, invert to get zb
        self.uvWSE_inversion = None
        #truth zb for inversion comparison
        self.zb_truth = None

        #min and max of variables for training (for scaling purpuse)
        self.variables_min_max = None

    def parse_flow_data(self, serialized_example, bWithIBathy=False):
        features = {
            'iBathy': tf.io.FixedLenFeature([], tf.int64),
            'zb': tf.io.FixedLenFeature([], tf.string),
            'vel_WSE': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_single_example(serialized_example, features)

        iBathy = parsed_features['iBathy']

        zb = parsed_features['zb']  # get byte string
        zb = tf.io.parse_tensor(zb, out_type=tf.float64)  # restore 2D array from byte string

        vel_WSE = parsed_features['vel_WSE']  # get byte string
        vel_WSE = tf.io.parse_tensor(vel_WSE, out_type=tf.float64)  # restore 2D array from byte string

        if not bWithIBathy:  # don't return the iBathy data (not needed for training; cause error)
            return zb, vel_WSE
        else:
            return iBathy, zb, vel_WSE


    def load_training_validation_data(self):
        """Load training and validation data

        :return:
        """

        print("SWE2DDataLoader loading training and validation data ...")

        training_data = tf.data.TFRecordDataset(self.config.dataLoader.training_data)
        validation_data = tf.data.TFRecordDataset(self.config.dataLoader.validation_data)

        #count the number records for each data set
        self.nTraining_data = sum(1 for record in training_data)
        self.nValidation_data = sum(1 for record in validation_data)

        # calcualte how many batches in training, validation, and test
        self.nTraining_batches = int(self.nTraining_data / self.config.trainer.batch_size)
        self.nValidation_batches = int(self.nValidation_data / self.config.trainer.batch_size)

        # Transform binary data into image arrays
        training_data = training_data.map(self.parse_flow_data)
        validation_data = validation_data.map(self.parse_flow_data)

        training_dataset = training_data.shuffle(buffer_size=512)
        training_dataset = training_dataset.batch(self.config.trainer.batch_size, drop_remainder=True)
        training_dataset = training_dataset.repeat()

        validation_dataset = validation_data.shuffle(buffer_size=512)
        validation_dataset = validation_dataset.batch(self.config.trainer.batch_size, drop_remainder=True)
        validation_dataset = validation_dataset.repeat()

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

    def load_test_data(self):
        """Load test data

        :return:
        """

        print("SWE2DDataLoader loading test data ...")

        test_data = tf.data.TFRecordDataset(self.config.dataLoader.test_data)

        # count the number records for each data set
        self.nTest_data = sum(1 for record in test_data)

        # calcualte how many batches in test
        self.nTest_batches = int(self.nTest_data / self.config.trainer.batch_size)

        # Transform binary data into image arrays
        # test data also include the iBathy ID
        test_data = test_data.map(lambda x: self.parse_flow_data(x, bWithIBathy=True))

        self.test_dataset = test_data

    def calculate_input_output_shape(self):
        """
        Calcualte the shapes of input and output. Using test data to do the calculation.

        :return:
        """

        if self.training_dataset == None:
            self.load_training_validation_data()

        # Create an iterator for reading a batch of input and output test data
        iterator = iter(self.training_dataset)
        zb, vel_WSE = next(iterator)

        #zb is of the shape for example [32, 64, 256, 1]. We don't need the first element, which
        #is the batch size.
        input_shape = zb.shape.as_list()
        input_shape.pop(0)
        output_shape = vel_WSE.shape.as_list()
        output_shape.pop(0)

        self.input_data_shape = input_shape
        self.output_data_shape = output_shape

    def load_inversion_data(self):
        """Load inversion (if specified in the config file)

        :return:
        """

        self.uvWSE_inversion = np.load(self.config.inverter.inversion_data_files)['uvWSE']
        self.zb_truth = np.load(self.config.inverter.inversion_data_files)['zb']

    def load_variables_min_max(self):
        """
        Load min and max for variables (for scaling purpose)

        :return:
        """

        # load the JSON file
        with open(self.config.dataLoader.minMaxVars_file) as json_file:
            self.variables_min_max = json.load(json_file)

        #check
        bounds = self.variables_min_max['bounds']
        print("bounds = ", bounds)

    def get_variables_min_max(self):
        if self.variables_min_max is None:
            self.load_variables_min_max()

        return self.variables_min_max

    def get_training_data(self):
        if self.training_dataset == None:
            self.load_training_validation_data()

        return self.training_dataset

    def get_validation_data(self):
        if self.validation_dataset == None:
            self.load_training_validation_data()

        return self.validation_dataset

    def get_test_data(self):
        if self.test_dataset == None:
            self.load_test_data()

        return self.test_dataset

    def get_nTraining_data(self):
        if self.nTraining_data == -1:
            self.load_training_validation_data()

        return self.nTraining_data

    def get_nValidation_data(self):
        if self.nValidation_data == -1:
            self.load_training_validation_data()

        return self.nValidation_data

    def get_nTest_data(self):
        if self.nTest_data == -1:
            self.load_test_data()

        return self.nTest_data

    def get_nTraining_batches(self):
        if self.nTraining_batches == -1:
            self.load_training_validation_data()

        return self.nTraining_batches

    def get_nValidation_batches(self):
        if self.nValidation_batches == -1:
            self.load_training_validation_data()

        return self.nValidation_batches

    def get_nTest_batches(self):
        if self.nTest_batches == -1:
            self.load_training_validation_data()

        return self.nTest_batches

    def get_input_data_shape(self):
        if self.input_data_shape == None:
            self.calculate_input_output_shape()

        return self.input_data_shape

    def get_output_data_shape(self):
        if self.output_data_shape == None:
            self.calculate_input_output_shape()

        return self.output_data_shape

    def get_uvWSE_inversion(self):
        if self.uvWSE_inversion is None:
            self.load_inversion_data()

        return self.uvWSE_inversion

    def get_zb_truth(self):
        if self.zb_truth is None:
            self.load_inversion_data()

        return self.zb_truth
