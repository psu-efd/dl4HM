from ..base.base_inverter import BaseInverter
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as kb

class SWEs2DModelInverter(BaseInverter):
    def __init__(self, modelWrapper, dataLoader, config):
        super(SWEs2DModelInverter, self).__init__(modelWrapper, dataLoader, config)

        #a random number generator for perturbation
        self.rand_gen = tf.random.Generator.from_seed(1)

        #shape of input (zb) and output (uvWSE) of the NN surrogate model
        self.input_data_shape = dataLoader.get_input_data_shape()
        self.output_data_shape = dataLoader.get_output_data_shape()

        #load target uvWSE (called original because we may extract only u,v for inversion
        self.uvWSE_target_np_org = dataLoader.get_uvWSE_inversion()

        #check whether to use (u,v) only or (u,v,WSE) to do the inversion.
        #If the inversion data (self.uvWSE_target_np) only contains (u,v),
        #then do nothing; otherwise need to extract (u,v) if desired
        self.b_need_to_extract_uv = False
        if "use_uv_only" in config.inverter:
            if config.inverter.use_uv_only and (self.uvWSE_target_np_org.shape[-1] == 3):
                print("Surrogate model output contains (u,v,WSE). But the inverter only uses (u,v).")
                self.b_need_to_extract_uv = True

        if self.b_need_to_extract_uv:
            self.uvWSE_target_np = self.uvWSE_target_np_org[:,:,0:2]   #extract only u and v.
        else:
            self.uvWSE_target_np = self.uvWSE_target_np_org

        self.uvWSE_target_np = self.uvWSE_target_np[np.newaxis, :, :, :]   #expand one more dimension (not necessary?)
        self.uvWSE_target = tf.Variable(self.uvWSE_target_np, trainable=False, name="uvWSE_target", dtype=np.float32)
        self.uvWSE_target_org = self.uvWSE_target    #a backup in case we need to add random perturbations to uvWSE_target

        self.uvWSE_pred = None

        # whether to use masks for inversion
        self.b_use_masks = False  # default
        self.masks_np_org = None
        self.masks_np = None
        self.masks = None

        # default masks are just 1.0
        self.masks_np_org = np.ones(shape=self.uvWSE_target_np.shape)
        self.masks_np = np.ones(shape=self.uvWSE_target_np.shape)

        if "b_use_masks" in config.inverter:
            if config.inverter['b_use_masks']:
                self.b_use_masks = True

                if "mask_filename" not in config.inverter:
                    raise Exception("b_use_masks is set to true, but mask_filename is not set.")
                else:
                    mask_filename = config.inverter['mask_filename']
                    self.masks_np_org = np.load(mask_filename)['masks']

                    # make the masks the same shape as uvWSE_target and uvWSE_pred
                    if "use_uv_only" in config.inverter:  # only need u and v
                        self.masks_np = np.repeat(self.masks_np_org[:, :, np.newaxis], 2, axis=2)
                    else: #use u, v, and WSE
                        self.masks_np = np.repeat(self.masks_np_org[:, :, np.newaxis], 3, axis=2)

                    self.masks_np = self.masks_np[np.newaxis, :, :, :]  # expand one more dimension

        self.masks = tf.Variable(self.masks_np, trainable=False, name="masks", dtype=np.float32)

        #a multiplication factor for value loss due to masking. This is because the masking will reduce the
        #value loss magnitude. To have the same comparison with other losses, we need this multiplication factor
        self.masks_multiplication_factor = (self.output_data_shape[0]*self.output_data_shape[1])/(self.masks_np_org.sum()/self.masks_np_org.shape[-1])
        print("In inverter: masks_multiplication_factor = ", self.masks_multiplication_factor)
        print("In inverter: using ", self.masks_np_org.sum()/self.masks_np_org.shape[-1], " subset out of ",  (self.output_data_shape[0]*self.output_data_shape[1]))

        #the inversion variable, i.e., zb
        #self.zb_np = np.zeros(self.input_data_shape)
        self.zb_np = (np.random.random(self.input_data_shape)-0.5)*0.01

        #expand one dimension to zb_np, e.g, [64, 256, 1] to [1, 64, 256, 1]
        self.zb_np = self.zb_np[np.newaxis, :, :, :]

        self.zb = tf.Variable(self.zb_np, trainable=True, name="zb", dtype=np.float32)

        # Is the tape that computes the gradients!
        #self.trainable_variables = self.zb

        # Regularizer
        #self.regularizer = tf.keras.regularizers.L2(self.config.inverter.L2_regularization_factor)

        # Optimizer
        self.optimizer = None
        if self.config.inverter.optimizer == "adam":
            self.optimizer = tf.optimizers.Adam(learning_rate=self.config.inverter.adam.learning_rate,
                                                epsilon=self.config.inverter.adam.epsilon)
        elif self.config.inverter.optimizer == "SGD":
            self.optimizer = tf.optimizers.SGD(learning_rate=self.config.inverter.learning_rate, momentum=0.9)
        else:
            raise Exception("Specified inversion optimizer not recognized.")

        # loss (error)
        self.loss = 0.0

        # record the loss history
        self.losses = []

    def initialize_variables(self, zb_init=None):
        """
        (Re-)Initialize variables before each inversion.

        Inversion can be done repeatedly with different initial guess on zb

        """

        if zb_init is None:
            self.zb_np = (np.random.random(self.input_data_shape) - 0.5) * 0.0
        else:
            #make sure the provided zb_np is in the correct shape, e.g., [64, 256, 1]
            assert zb_init.shape[0] == self.input_data_shape[0] and \
                zb_init.shape[1] == self.input_data_shape[1]

            self.zb_np = zb_init

        # expand one dimension to zb_np, e.g, [64, 256, 1] to [1, 64, 256, 1]
        self.zb_np = self.zb_np[np.newaxis, :, :, :]

        self.zb = tf.Variable(self.zb_np, trainable=True, name="zb", dtype=np.float32)

        self.loss = 0.0

        self.losses = []

        # intermediate inverted zb
        self.zb_intermediate = None


    def invert(self,bSave_intermediate=False):
        """
        Perform inversion to get the bed

        :return:
        """

        for i in range(self.config.inverter.nSteps):
            #save intermediate zb
            if i == 0:
                self.zb_intermediate = np.squeeze(self.zb.numpy()[0])
            else:
                self.zb_intermediate = np.dstack((self.zb_intermediate, np.squeeze(self.zb.numpy()[0])))

            with tf.GradientTape() as tape:
                # use the surrogate NN model to make a prediction
                self.uvWSE_pred = self.modelWrapper.model(self.zb, training=False)

                # difference between target and predicted WSE values
                if self.b_need_to_extract_uv:
                    loss_prediction_error = tf.math.reduce_sum(
                        tf.math.multiply(self.masks,
                                         tf.math.squared_difference(self.uvWSE_target, self.uvWSE_pred[:,:,:,0:2])
                                         )
                    )
                    #loss_prediction_error = tf.math.reduce_mean(tf.math.squared_difference(self.uvWSE_target, self.uvWSE_pred[:,:,:,0:2]))
                else:
                    loss_prediction_error = tf.math.reduce_sum(
                        tf.math.multiply(self.masks,
                                         tf.math.squared_difference(self.uvWSE_target, self.uvWSE_pred)
                                         )
                    )
                    # loss_prediction_error = tf.math.reduce_mean(tf.math.squared_difference(self.uvWSE_target, self.uvWSE_pred))

                #multiplication factor due to masking
                loss_prediction_error *= self.masks_multiplication_factor

                self.loss = loss_prediction_error

                # add value regularization
                #loss_value_regularization = self.regularizer(self.zb)
                loss_value_regularization = self.config.inverter.value_regularization_factor * \
                                            tf.math.reduce_sum(tf.nn.relu(tf.math.abs(self.zb - self.config.inverter.value_regularization_mean) -
                                                                          self.config.inverter.value_regularization_amplitude))

                self.loss += loss_value_regularization

                # use gradient of zb (slope) as the regularizer (to favor smoother solutions)
                dzb_dx = tf.math.abs(tf.experimental.numpy.diff(self.zb, axis=2))
                dzb_dy = tf.math.abs(tf.experimental.numpy.diff(self.zb, axis=1))
                #loss_slope = self.config.inverter.slope_regularization_factor * (tf.math.reduce_mean(dzb_dx) + tf.math.reduce_mean(dzb_dy))
                loss_slope = self.config.inverter.slope_regularization_factor * (
                            tf.math.reduce_sum(tf.nn.relu(tf.math.abs(dzb_dx - self.config.inverter.slope_regularization_mean_xslope) -
                                                          self.config.inverter.slope_regularization_amplitude)) +
                            tf.math.reduce_sum(tf.nn.relu(tf.math.abs(dzb_dy - self.config.inverter.slope_regularization_mean_yslope) -
                                                          self.config.inverter.slope_regularization_amplitude))
                )

                self.loss += loss_slope

                print("Iter. #, total loss, prediction error loss, value regularization loss, slope loss: ",
                      i, self.loss.numpy(), loss_prediction_error.numpy(), loss_value_regularization.numpy(), loss_slope.numpy())

                self.losses.append([self.loss.numpy(), loss_prediction_error.numpy(), loss_value_regularization.numpy(), loss_slope.numpy()])

            grads = tape.gradient(self.loss, [self.zb])

            self.optimizer.apply_gradients(zip(grads, [self.zb]))

        #save the loss history to file
        #np.savez("inversion_loss_history.npz", loss=self.losses)

    def perturb_uvWSE(self, perturb_level):
        """
        Add some perturbation to (u,v,WSE)

        :return:
        """

        perturb = self.rand_gen.uniform(shape=self.uvWSE_target_org.shape, minval=-perturb_level, maxval=perturb_level)

        self.uvWSE_target = tf.math.add(self.uvWSE_target_org, perturb)

    def get_input_data_shape(self):
        return self.input_data_shape

    def get_zb(self):
        return self.zb.numpy()[0]

    def get_uvWSE_pred(self):
        return self.uvWSE_pred.numpy()[0]

    def get_uvWSE_target(self):
        #return self.uvWSE_target_np_org
        return self.uvWSE_target.numpy()[0]

    def get_inversion_loss_history(self):
        return np.array(self.losses)

    def get_zb_intermediate(self):
        return self.zb_intermediate
