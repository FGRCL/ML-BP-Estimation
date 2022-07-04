from abc import ABC, abstractmethod
from typing import Callable, Any

from numpy import ndarray
from tensorflow import Tensor, numpy_function, DType
from tensorflow.python.data import Dataset


class DatasetOperations(ABC):
    def __init__(self, *additional_args):
        self.additional_args = additional_args

    @abstractmethod
    def apply(self, dataset: Dataset) -> Dataset:
        ...


class TransformOperation(DatasetOperations):
    def __init__(self, transform: Callable[[Tensor, Tensor, ...], Tensor], *additional_arguments):
        super().__init__(*additional_arguments)
        self.transform = transform
        self.additional_arguments = additional_arguments

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.transform_func)

    def transform_func(self, x: Tensor, y: Tensor = None):
        if y is not None:
            return self.transform(x, y, *self.additional_args)
        else:
            return self.transform(x, *self.additional_args)


class FilterOperation(DatasetOperations):
    def __init__(self, _filter: Callable[[Tensor, Tensor, ...], bool], *additional_args):
        super().__init__(*additional_args)
        self.filter = _filter

    def apply(self, dataset) -> Dataset:
        return dataset.filter(self.filter_func)

    def filter_func(self, x: Tensor, y: Tensor = None):
        if y is not None:
            return self.filter(x, y, *self.additional_args)
        else:
            return self.filter(x, *self.additional_args)


class NumpyTransformOperation(DatasetOperations):
    def __init__(self, transform: Callable[[ndarray, ...], Any], out_type: DType, *additional_args):
        super().__init__(*additional_args)
        self.transform = transform
        self.out_type = out_type

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.adapted_function)

    def adapted_function(self, x: Tensor, y: Tensor = None):
        if y is not None:
            return numpy_function(self.transform, [x, y, *self.additional_args], self.out_type)
        else:
            return numpy_function(self.transform, [x, *self.additional_args], self.out_type)


class DatasetPreprocessingPipeline(ABC):
    def __init__(self, dataset_operations: list[DatasetOperations]):
        self.dataset_operations = dataset_operations

    def preprocess(self, dataset: Dataset) -> Dataset:
        for op in self.dataset_operations:
            dataset = op.apply(dataset)
        return dataset
