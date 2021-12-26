from ..base.base_data_loader import BaseDataLoader
import numpy as np
import json

import tensorflow as tf


class SWEs2DDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SWEs2DDataLoader, self).__init__(config)

        #the total number of records in training, validation, and test data sets
        self.nTraining_data = 0
        self.nValidation_data = 0
        self.nTest_data = 0

        #load data
        self.training_dataset, self.validation_dataset, self.test_dataset = self.load_data()

        #calcualte how many batches in training, validation, and test
        self.train_batches = int(self.nTraining_data/self.config.trainer.batch_size)
        self.validation_batches = int(self.nValidation_data/self.config.trainer.batch_size)
        self.test_batches = int(self.nTest_data/self.config.trainer.batch_size)

        #sizes of input and output
        #input: zb_bed
        #output: (u, v, WSE)
        self.input_data_shape, self.output_data_shape = self.calculate_input_output_shape()

    def load_data(self):
        """Load training and test data

        :return:
        """

        training_data = tf.data.TFRecordDataset(self.config.dataLoader.training_data)
        validation_data = tf.data.TFRecordDataset(self.config.dataLoader.validation_data)
        test_data = tf.data.TFRecordDataset(self.config.dataLoader.test_data)

        #count the number records for each data set
        self.nTraining_data = sum(1 for record in training_data)
        self.nValidation_data = sum(1 for record in validation_data)
        self.nTest_data = sum(1 for record in test_data)

        def parse_flow_data(serialized_example):
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

            #don't return the iBathy data (not needed for training; cause error)
            return zb, vel_WSE

        # Transform binary data into image arrays
        training_data = training_data.map(parse_flow_data)
        validation_data = validation_data.map(parse_flow_data)
        test_data = test_data.map(parse_flow_data)

        training_dataset = training_data.shuffle(buffer_size=512)
        training_dataset = training_dataset.batch(self.config.trainer.batch_size, drop_remainder=True)
        training_dataset = training_dataset.repeat()

        validation_dataset = validation_data.shuffle(buffer_size=512)
        validation_dataset = validation_dataset.batch(self.config.trainer.batch_size, drop_remainder=True)
        validation_dataset = validation_dataset.repeat()

        test_dataset = test_data.batch(self.config.trainer.batch_size, drop_remainder=True)
        test_dataset = test_dataset.repeat()

        return training_dataset, validation_dataset, test_dataset


    def calculate_input_output_shape(self):
        """
        Calcualte the shapes of input and output. Using test data to do the calculation.

        :return:
        """

        # Create an iterator for reading a batch of input and output test data
        iterator = iter(self.training_dataset)
        zb, vel_WSE = next(iterator)

        #zb is of the shape for example [32, 21, 101, 1]. We don't need the first element, which
        #is the batch size.
        input_shape = zb.shape.as_list()
        input_shape.pop(0)
        output_shape = vel_WSE.shape.as_list()
        output_shape.pop(0)

        return input_shape, output_shape

    def load_inversion_data(self):
        """Load inversion (if specified in the config file)

        :return:
        """

        if self.config.inverter.do_inversion:
            self.WSE_inversion = np.load(self.config.inverter.inversion_data)['WSE']
            self.zb_beds_truth = np.load(self.config.inverter.inversion_data)['zb_beds']
        else:
            self.WSE_inversion = None

    def get_train_data(self):
        return self.training_dataset

    def get_validation_data(self):
        return self.validation_dataset

    def get_test_data(self):
        return self.test_dataset

    def get_WSE_inversion(self):
        return self.WSE_inversion

    def get_zb_beds_truth(self):
        return self.zb_beds_truth
