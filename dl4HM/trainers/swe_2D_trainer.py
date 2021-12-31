from ..base.base_trainer import BaseTrainer
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
import json

from ..utils.misc import NumpyFloatValuesEncoder

# Implement callback function to record loss components
class myCallback(tf.keras.callbacks.Callback):
    """my callback function with additional data passed in.

    """

    def __init__(self, trainer):
        """ Pass additional data in constructor

        :argument
            trainer: FunctionFittingModelTrainer


        """
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs={}):
        self.trainer.modelWrapper.loss_value_epoch.append(self.trainer.modelWrapper.loss_value[-1])

class SWEs2DModelTrainer(BaseTrainer):
    def __init__(self, modelWrapper, dataLoader, config):
        super(SWEs2DModelTrainer, self).__init__(modelWrapper, dataLoader, config)

        self.callbacks = []
        self.history = None
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        #add an early stopping callback
        #self.callbacks.append(
        #    tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',
        #                                     patience=10, restore_best_weights=True)
        #)

        # add adaptive learning rate schedule callback
        self.callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=self.config.callbacks.ReduceLROnPlateau_monitor,
                factor=self.config.callbacks.ReduceLROnPlateau_factor,
                patience=self.config.callbacks.ReduceLROnPlateau_patience,
                min_lr=self.config.callbacks.ReduceLROnPlateau_min_lr,
                verbose=1
            )
        )

        # add model check point callback
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                      '%s-{epoch:02d}.hdf5' % self.config.case.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        # add TensorBoard callback
        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        #add loss components recorder
        self.callbacks.append(myCallback(self))

    def train(self):
        if self.modelWrapper.model is None:
            raise Exception("The model does not exist yet. "
                            "You have to call modelWrapper's build_model() or load().")

        #First we check whether the input_shape in the config file is the same as the shape of the training data
        if self.config.model.input_shape[0] != self.dataLoader.get_input_data_shape()[0] or \
            self.config.model.input_shape[1] != self.dataLoader.get_input_data_shape()[1]:
            raise Exception("The input_shape in the config file does not match with the shape of training data. "
                            "input_shape = ", self.config.model.input_shape,
                            "training data shape = ", self.dataLoader.get_input_data_shape())

        self.history = self.modelWrapper.model.fit(
            self.dataLoader.get_training_data(),
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch = self.dataLoader.get_nTraining_batches(),
            validation_data=self.dataLoader.get_validation_data(),
            validation_steps=self.dataLoader.get_nValidation_batches(),
            verbose=self.config.trainer.verbose_training,
            callbacks=self.callbacks
        )

        if 'loss' in self.history.history:
            self.loss.extend(self.history.history['loss'])

        if 'acc' in self.history.history:
            self.acc.extend(self.history.history['acc'])

        if 'val_loss' in self.history.history:
            self.val_loss.extend(self.history.history['val_loss'])

        if 'val_acc' in self.history.history:
            self.val_acc.extend(self.history.history['val_acc'])

        #save the training history to a JSON file for later use
        #print(self.history.history)
        with open(self.config.trainer.history_save_filename, "w") as history_file:
            json.dump(self.history.history, history_file, indent=4, cls=NumpyFloatValuesEncoder)