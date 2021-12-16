class BaseModelWrapper(object):
    """A wrapper for TF's Model

    """
    def __init__(self, config, dataLoader):
        self.config = config
        self.dataLoader =  dataLoader
        self.model = None #This is the real TF's Model

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError

    def test_model(self, test_data, verbose=0):
        raise NotImplementedError

    def predict(self, input_data, verbose=0):
        raise NotImplementedError
