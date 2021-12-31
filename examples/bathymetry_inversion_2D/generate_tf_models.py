"""
Generate Tensorflow NN models and save to JSON files, which can be used by other code.

"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, UpSampling2D, Conv2DTranspose, Input, Conv2D, Flatten, Reshape
from tensorflow.keras import layers

from tensorflow.keras import backend as kb
import tensorflow as tf

import numpy as np

import json



def generate_tf_model_64_by_256(b_uv_only):
    """
    Generate and save Tensorflow NN model.

    :param b_uv_only: bool
        whether it s uv_only or (u,v,WSE)

    :return:
    """

    def conv(input):
        """"
        Define layers to calculate the convolution and FC part of the network

        """

        # Set the number of filters for the first convolutional layer
        x = layers.Conv2D(128, (16, 16), strides=(16, 16), padding='same', name='conv1', activation='relu')(input)

        # Set the number of filters and kernel size for the second convolutional layer
        x = layers.Conv2D(512, (4, 4), strides=(4, 4), padding='same', name='conv2', activation='relu')(x)

        x = layers.Flatten()(x)

        ### Add a denslayer with ReLU activation
        x = layers.Dense(1024, activation='relu')(x)
        ###

        # Reshape the output as 1x1 image with xxx channels:
        x = layers.Reshape((1, 1, 1024))(x)

        return x

    def deconv(input, suffix):
        """
        Define layers that perform the deconvolution steps

        """

        x = layers.Conv2DTranspose(512, (8, 8), strides=(8, 8), activation='relu', name="deconv1_" + suffix)(input)

        # Add the 2nd and 3rd Conv2DTranspose layers
        x = layers.Conv2DTranspose(256, (8, 2), strides=(8, 2), activation='relu', name="deconv2_" + suffix)(x)
        x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', name="deconv3_" + suffix)(x)

        # For the last layer, the activation function should match with the output value range. For example, we
        # can't use "relu" if the output range covers the negative side, e.g., [-0.5, 0.5]
        x = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), activation='linear', name="deconv4_" + suffix)(x)
        x = layers.Permute((2, 1, 3), name="permute_" + suffix)(x)

        return x

    def conv_deconv(input):
        # Combine the convolution / deconvolution steps

        x = conv(input)

        # Add decoder for u
        u = deconv(x, "u")

        # Add decoder for v
        v = deconv(x, "v")

        # Initialize NN output
        output_NN = None

        if b_uv_only:  # The NN output only includes u and v
            output_NN = layers.concatenate([u, v], axis=3)
        else:  # The NN output includes u, v, and WSE
            # Add decoder for WSE
            WSE = deconv(x, "WSE")
            output_NN = layers.concatenate([u, v, WSE], axis=3)

        return output_NN

    input = Input(shape=(64, 256, 1), name="bathymetry")

    output = conv_deconv(input)

    if b_uv_only:
        model_name = "model_64_by_256_uv_only"
    else:
        model_name = "model_64_by_256_uvWSE"

    conv_model = tf.keras.Model(inputs=input, outputs=output, name=model_name)

    conv_model.summary()

    # save the model to JSON file
    model_config = conv_model.to_json()

    if b_uv_only:
        with open("model_64_by_256_uv_only.json", 'w') as json_file:
            json.dump(model_config, json_file, indent=4)
    else:
        with open("model_64_by_256_uvWSE.json", 'w') as json_file:
            json.dump(model_config, json_file, indent=4)

    # read the model back from the JSON file
    #with open("model.json", 'r') as json_file:
    #    model_config_new = json.load(json_file)

    #loaded_model = tf.keras.models.model_from_json(model_config_new)
    #loaded_model.summary()


def generate_tf_model_16_by_64(b_uv_only):
    """
    Generate and save Tensorflow NN model.

    :param b_uv_only: bool
        whether it s uv_only or (u,v,WSE)

    :return:
    """

    def conv(input):
        """"
        Define layers to calculate the convolution and FC part of the network

        """

        # Set the number of filters for the first convolutional layer
        x = layers.Conv2D(128, (8, 8), strides=(8, 8), padding='same', name='conv1', activation='relu')(input)

        # Set the number of filters and kernel size for the second convolutional layer
        x = layers.Conv2D(512, (4, 4), strides=(4, 4), padding='same', name='conv2', activation='relu')(x)

        x = layers.Flatten()(x)

        ### Add a denslayer with ReLU activation
        x = layers.Dense(1024, activation='relu')(x)
        ###

        # Reshape the output as 1x1 image with xxx channels:
        x = layers.Reshape((1, 1, 1024))(x)

        return x

    def deconv(input, suffix):
        """
        Define layers that perform the deconvolution steps

        """

        x = layers.Conv2DTranspose(512, (4, 4), strides=(4, 4), activation='relu', name="deconv1_" + suffix)(input)

        # Add the 2nd and 3rd Conv2DTranspose layers
        x = layers.Conv2DTranspose(256, (8, 2), strides=(8, 2), activation='relu', name="deconv2_" + suffix)(x)
        #x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', name="deconv3_" + suffix)(x)

        # For the last layer, the activation function should match with the output value range. For example, we
        # can't use "relu" if the output range covers the negative side, e.g., [-0.5, 0.5]
        x = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), activation='linear', name="deconv4_" + suffix)(x)
        x = layers.Permute((2, 1, 3), name="permute_" + suffix)(x)

        return x

    def conv_deconv(input):
        # Combine the convolution / deconvolution steps

        x = conv(input)

        # Add decoder for u
        u = deconv(x, "u")

        # Add decoder for v
        v = deconv(x, "v")

        # Initialize NN output
        output_NN = None

        if b_uv_only:  # The NN output only includes u and v
            output_NN = layers.concatenate([u, v], axis=3)
        else:  # The NN output includes u, v, and WSE
            # Add decoder for WSE
            WSE = deconv(x, "WSE")
            output_NN = layers.concatenate([u, v, WSE], axis=3)

        return output_NN

    input = Input(shape=(16, 64, 1), name="bathymetry")

    output = conv_deconv(input)

    if b_uv_only:
        model_name = "model_16_by_64_uv_only"
    else:
        model_name = "model_16_by_64_uvWSE"

    conv_model = tf.keras.Model(inputs=input, outputs=output, name=model_name)

    conv_model.summary()

    # save the model to JSON file
    model_config = conv_model.to_json()

    if b_uv_only:
        with open("model_16_by_64_uv_only.json", 'w') as json_file:
            json.dump(model_config, json_file, indent=4)
    else:
        with open("model_16_by_64_uvWSE.json", 'w') as json_file:
            json.dump(model_config, json_file, indent=4)


def generate_tf_model_8_by_32(b_uv_only):
    """
    Generate and save Tensorflow NN model.

    :param b_uv_only: bool
        whether it s uv_only or (u,v,WSE)

    :return:
    """

    def conv(input):
        """"
        Define layers to calculate the convolution and FC part of the network

        """

        # Set the number of filters for the first convolutional layer
        x = layers.Conv2D(128, (4, 4), strides=(4, 4), padding='same', name='conv1', activation='relu')(input)

        # Set the number of filters and kernel size for the second convolutional layer
        x = layers.Conv2D(512, (2, 2), strides=(2, 2), padding='same', name='conv2', activation='relu')(x)

        x = layers.Flatten()(x)

        ### Add a denslayer with ReLU activation
        x = layers.Dense(512, activation='relu')(x)
        ###

        # Reshape the output as 1x1 image with xxx channels:
        x = layers.Reshape((1, 1, 512))(x)

        return x

    def deconv(input, suffix):
        """
        Define layers that perform the deconvolution steps

        """

        x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', name="deconv1_" + suffix)(input)

        # Add the 2nd and 3rd Conv2DTranspose layers
        x = layers.Conv2DTranspose(128, (8, 2), strides=(8, 2), activation='relu', name="deconv2_" + suffix)(x)
        #x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', name="deconv3_" + suffix)(x)

        # For the last layer, the activation function should match with the output value range. For example, we
        # can't use "relu" if the output range covers the negative side, e.g., [-0.5, 0.5]
        x = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), activation='linear', name="deconv4_" + suffix)(x)
        x = layers.Permute((2, 1, 3), name="permute_" + suffix)(x)

        return x

    def conv_deconv(input):
        # Combine the convolution / deconvolution steps

        x = conv(input)

        # Add decoder for u
        u = deconv(x, "u")

        # Add decoder for v
        v = deconv(x, "v")

        # Initialize NN output
        output_NN = None

        if b_uv_only:  # The NN output only includes u and v
            output_NN = layers.concatenate([u, v], axis=3)
        else:  # The NN output includes u, v, and WSE
            # Add decoder for WSE
            WSE = deconv(x, "WSE")
            output_NN = layers.concatenate([u, v, WSE], axis=3)

        return output_NN

    input = Input(shape=(8, 32, 1), name="bathymetry")

    output = conv_deconv(input)

    if b_uv_only:
        model_name = "model_8_by_32_uv_only"
    else:
        model_name = "model_8_by_32_uvWSE"

    conv_model = tf.keras.Model(inputs=input, outputs=output, name=model_name)

    conv_model.summary()

    # save the model to JSON file
    model_config = conv_model.to_json()

    if b_uv_only:
        with open("model_8_by_32_uv_only.json", 'w') as json_file:
            json.dump(model_config, json_file, indent=4)
    else:
        with open("model_8_by_32_uvWSE.json", 'w') as json_file:
            json.dump(model_config, json_file, indent=4)


if __name__ == '__main__':

    #generate and save Tensorflow NN model which is suitable for:
    #Input: [:,64,256,1]
    #Output: []
    #generate_tf_model_64_by_256(b_uv_only=True)  #generate uv_only, i.e., output = (u, v)
    #generate_tf_model_64_by_256(b_uv_only=False)  # generate uvWSE, i.e., output = (u,v, WSE)

    #generate_tf_model_16_by_64(b_uv_only=True)  #generate uv_only, i.e., output = (u, v)
    #generate_tf_model_16_by_64(b_uv_only=False)  # generate uvWSE, i.e., output = (u,v, WSE)

    generate_tf_model_8_by_32(b_uv_only=True)  #generate uv_only, i.e., output = (u, v)
    generate_tf_model_8_by_32(b_uv_only=False)  # generate uvWSE, i.e., output = (u,v, WSE)

    print("All done!")