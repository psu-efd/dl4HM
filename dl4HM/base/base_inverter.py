class BaseInverter(object):
    def __init__(self, modelWrapper, dataLoader, config):
        self.modelWrapper = modelWrapper
        self.dataLoader = dataLoader
        self.config = config

    def train(self):
        raise NotImplementedError