from time import time

from keras.callbacks import Callback


class EpochTime(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time() - self.epoch_start_time
        logs['epoch time'] = epoch_end_time
