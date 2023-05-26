from abc import abstractmethod

from keras import Model


class BloodPressureModel(Model):

    @abstractmethod
    def set_input_shape(self, dataset_spec):
        pass
