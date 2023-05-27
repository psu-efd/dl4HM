"""
Generate Tensorflow NN models and save to JSON files, which can be used later, e.g., load in by swe_2D_model.

"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, UpSampling2D, Conv2DTranspose, Input, Conv2D, Flatten, Reshape
from tensorflow.keras import layers

import tensorflow as tf

import json

def generate_tf_model_32_by_128(b_uv_only):
    """
    Generate and save Tensorflow NN model.

    Note: Regularizaiton parameter is hard-coded.

    :param b_uv_only: bool
        whether it s uv_only or (u,v,WSE)

    :return:
    """

    # Regularization parameter
    lamda = 2e-7

    def conv(input):
        """"
        Define layers to calculate the convolution and FC part of the network

        """

        # Set the number of filters for the first convolutional layer
        x = layers.Conv2D(128, (8, 8), strides=(8, 8), padding='same', name='conv1',
                          activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda))(input)

        # Set the number of filters and kernel size for the second convolutional layer
        x = layers.Conv2D(512, (4, 4), strides=(4, 4), padding='same', name='conv2',
                          activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda))(x)

        x = layers.Flatten()(x)

        ### Add a denslayer with ReLU activation
        x = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamda))(x)
        ###

        # Reshape the output as 1x1 image with xxx channels:
        x = layers.Reshape((1, 1, 1024))(x)

        return x

    def deconv(input, suffix):
        """
        Define layers that perform the deconvolution steps

        """

        x = layers.Conv2DTranspose(512, (8, 8), strides=(8, 8), activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(lamda), name="deconv1_" + suffix)(input)

        # Add the 2nd and 3rd Conv2DTranspose layers
        x = layers.Conv2DTranspose(256, (8, 2), strides=(8, 2), activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(lamda), name="deconv2_" + suffix)(x)
        #x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu',
        #                           kernel_regularizer=tf.keras.regularizers.l2(lamda), name="deconv3_" + suffix)(x)

        # For the last layer, the activation function should match with the output value range. For example, we
        # can't use "relu" if the output range covers the negative side, e.g., [-0.5, 0.5]
        x = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), activation='linear',
                                   kernel_regularizer=tf.keras.regularizers.l2(lamda),name="deconv4_" + suffix)(x)
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

    input = Input(shape=(32, 128, 1), name="bathymetry")

    output = conv_deconv(input)

    if b_uv_only:
        model_name = "model_32_by_128_uv_only"
    else:
        model_name = "model_32_by_128_uvWSE"

    conv_model = tf.keras.Model(inputs=input, outputs=output, name=model_name)

    conv_model.summary()

    # save the model to JSON file
    model_config = conv_model.to_json()

    if b_uv_only:
        with open(model_name+".json", 'w') as json_file:
            json.dump(model_config, json_file, indent=4)
    else:
        with open(model_name+".json", 'w') as json_file:
            json.dump(model_config, json_file, indent=4)

if __name__ == '__main__':

    #generate and save Tensorflow NN model which is suitable for:
    #Input: [:,32,128,1]
    #Output: []

    #generate_tf_model_32_by_128(b_uv_only=True)  #generate uv_only, i.e., output = (u, v)
    generate_tf_model_32_by_128(b_uv_only=False)  # generate uvWSE, i.e., output = (u,v, WSE)

    print("All done!")