from abc import ABC, abstractmethod

from tensorflow.python.keras import Model


class BloodPressureModel(ABC, Model):

    @abstractmethod
    def set_input_shape(self, dataset_spec):
        pass
