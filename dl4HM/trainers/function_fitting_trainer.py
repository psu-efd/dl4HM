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
        self.trainer.modelWrapper.loss_derivative_epoch.append(self.trainer.modelWrapper.loss_derivative[-1])

class FunctionFittingModelTrainer(BaseTrainer):
    def __init__(self, modelWrapper, dataLoader, config):
        super(FunctionFittingModelTrainer, self).__init__(modelWrapper, dataLoader, config)

        self.x_train, self.y_train = dataLoader.get_train_data()

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
            self.x_train, self.y_train,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            #batch_size=self.config.trainer.batch_size,
            batch_size=len(self.x_train),  # use only one batch. batch_size = number of training data points
            validation_split=self.config.trainer.validation_split,
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