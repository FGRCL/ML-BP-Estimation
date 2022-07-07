from abc import ABC, abstractmethod
from typing import Any, Tuple

import tensorflow as tf
from numpy import ndarray
from tensorflow import Tensor, numpy_function, py_function, DType
from tensorflow.python.data import Dataset


class DatasetOperation(ABC):
    @abstractmethod
    def apply(self, dataset: Dataset) -> Dataset:
        ...


class TransformOperation(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.transform)

    @abstractmethod
    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        ...


class FilterOperation(DatasetOperation):
    def apply(self, dataset) -> Dataset:
        return dataset.filter(self.filter)

    @abstractmethod
    def filter(self, x: Tensor, y: Tensor = None) -> bool:
        ...


class NumpyTransformOperation(DatasetOperation):
    def __init__(self, out_type: DType | Tuple[DType]):
        self.out_type = out_type

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.adapted_function)

    def adapted_function(self, x: Tensor, y: Tensor = None):
        if y is None:
            return numpy_function(self.transform, [x], self.out_type)
        else:
            return numpy_function(self.transform, [x, y], self.out_type)

    @abstractmethod
    def transform(self, x: ndarray, y: ndarray = None) -> Any:
        ...


class PythonFunctionTransformOperation(DatasetOperation):
    def __init__(self, out_type: DType | Tuple[DType]):
        self.out_type = out_type

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.adapted_function)

    def adapted_function(self, x: Tensor, y: Tensor = None):
        if y is None:
            return py_function(self.transform, [x], self.out_type)
        else:
            return py_function(self.transform, [x, y], self.out_type)

    @abstractmethod
    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        ...


class NumpyFilterOperation(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.filter(self.adapted_function)

    def adapted_function(self, x: Tensor, y: Tensor = None):
        if y is None:
            return numpy_function(self.filter, [x], bool)
        else:
            return numpy_function(self.filter, [x, y], bool)

    @abstractmethod
    def filter(self, x: ndarray, y: ndarray = None) -> Any:
        ...


class Print(TransformOperation):
    def __init__(self, operation_name):
        self.operation_name = operation_name

    def transform(self, x: Tensor, y: Tensor = None) -> Any:
        tf.print(self.operation_name)
        tf.print("x:", x)
        if y is not None:
            tf.print("y:", y)
        return x, y


class DatasetPreprocessingPipeline(ABC):
    def __init__(self, dataset_operations: list[DatasetOperation], debug=False):
        self.dataset_operations = dataset_operations
        self.debug = debug

    def preprocess(self, dataset: Dataset) -> Dataset:
        for op in self.dataset_operations:
            dataset = op.apply(dataset)
            if self.debug:
                dataset = Print(op.__class__.__name__).apply(dataset)
        return dataset
