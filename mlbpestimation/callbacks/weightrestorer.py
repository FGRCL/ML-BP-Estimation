import enum

from keras.callbacks import Callback


class Mode(enum.Enum):
    MINIMIZE = 1
    MAXIMIZE = 2


class RestoreBestWeights(Callback):
    def __init__(self, monitor: str, mode: Mode):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_weights = None
        self.best_value = None

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs[self.monitor]
        if self.mode is Mode.MINIMIZE:
            if self.best_value is None or current_value < self.best_value:
                self.best_value = current_value
                self.best_weights = self.model.get_weights()
        elif self.mode is Mode.MAXIMIZE:
            if self.best_value is None or current_value > self.best_value:
                self.best_value = current_value
                self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
