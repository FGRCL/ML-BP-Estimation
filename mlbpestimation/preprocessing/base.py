from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import tensorflow as tf
from tensorflow import DType, Tensor, numpy_function, py_function
from tensorflow.python.data import AUTOTUNE, Dataset


class DatasetOperation(ABC):
    @abstractmethod
    def apply(self, dataset: Dataset) -> Dataset:
        ...


class TransformOperation(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.transform, num_parallel_calls=AUTOTUNE, deterministic=False, name=self.__class__.__name__)

    @abstractmethod
    def transform(self, *args) -> Any:
        ...


class FilterOperation(DatasetOperation):
    def apply(self, dataset) -> Dataset:
        return dataset.filter(self.filter, name=self.__class__.__name__)

    @abstractmethod
    def filter(self, *args) -> bool:
        ...


class NumpyTransformOperation(DatasetOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], stateful: bool = False):
        self.out_type = out_type
        self.stateful = stateful

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.adapted_function, num_parallel_calls=AUTOTUNE, deterministic=False, name=self.__class__.__name__)

    def adapted_function(self, x: Tensor, y: Tensor = None):
        if y is None:
            return numpy_function(self.transform, [x], self.out_type, self.stateful, name=self.__class__.__name__)
        else:
            return numpy_function(self.transform, [x, y], self.out_type, self.stateful, name=self.__class__.__name__)

    @abstractmethod
    def transform(self, *args) -> Any:
        ...


class PythonFunctionTransformOperation(DatasetOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]]):
        self.out_type = out_type

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.adapted_function, num_parallel_calls=AUTOTUNE, deterministic=False, name=self.__class__.__name__)

    def adapted_function(self, x: Tensor, y: Tensor = None):
        if y is None:
            return py_function(self.transform, [x], self.out_type, name=self.__class__.__name__)
        else:
            return py_function(self.transform, [x, y], self.out_type, name=self.__class__.__name__)

    @abstractmethod
    def transform(self, *args) -> Any:
        ...


class NumpyFilterOperation(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.filter(self.adapted_function, name=self.__class__.__name__)

    def adapted_function(self, x: Tensor, y: Tensor = None):
        if y is None:
            return numpy_function(self.filter, [x], bool, name=self.__class__.__name__)
        else:
            return numpy_function(self.filter, [x, y], bool, name=self.__class__.__name__)

    @abstractmethod
    def filter(self, *args) -> Any:
        ...


class Batch(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.batch(self.get_batch_size(), num_parallel_calls=AUTOTUNE, deterministic=False, name=self.__class__.__name__)

    @abstractmethod
    def get_batch_size(self):
        ...


class FlatMap(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.flat_map(self.flatten, name=self.__class__.__name__)

    @abstractmethod
    def flatten(self, *args) -> Tuple[Dataset, ...]:
        ...


# This class is mainly used for debugging
class Print(TransformOperation):
    def __init__(self, operation_name):
        self.operation_name = operation_name

    def transform(self, *args) -> Any:
        tf.print(self.operation_name)
        for i, tensor in enumerate(args):
            tf.print(f'{i}:', tensor)
        return args


class DatasetPreprocessingPipeline(DatasetOperation):
    def __init__(self, dataset_operations: List[DatasetOperation]):
        self.dataset_operations = dataset_operations

    def apply(self, dataset: Dataset) -> Dataset:
        for op in self.dataset_operations:
            try:
                dataset = op.apply(dataset)
            except Exception as e:
                raise PreprocessingException(op.__class__).with_traceback(e.__traceback__) from e

        return dataset


class PreprocessingException(BaseException):
    def __init__(self, operation_name):
        super().__init__(f'Encountered an exception with operation: {operation_name}')
