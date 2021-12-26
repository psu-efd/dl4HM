from ..base.base_trainer import BaseTrainer
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf

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

        self.training_data = dataLoader.get_train_data()
        self.validation_data = dataLoader.get_validation_data()

        self.callbacks = []
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
        history = self.modelWrapper.model.fit(
            self.training_data,
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch = self.dataLoader.train_batches,
            validation_data=self.validation_data,
            validation_steps=self.dataLoader.validation_batches,
            verbose=self.config.trainer.verbose_training,
            callbacks=self.callbacks
        )

        if 'loss' in history.history:
            self.loss.extend(history.history['loss'])

        if 'acc' in history.history:
            self.acc.extend(history.history['acc'])

        if 'val_loss' in history.history:
            self.val_loss.extend(history.history['val_loss'])

        if 'val_acc' in history.history:
            self.val_acc.extend(history.history['val_acc'])