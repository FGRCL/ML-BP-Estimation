import wandb
from keras.callbacks import Callback


class EvaluateCallback(Callback):
    def on_test_end(self, logs=None):
        wandb.run.summary.update(
            logs
        )
