from os.path import splitext
from pathlib import Path
from re import match
from typing import Any, Tuple, Union

from numpy import split
from numpy.random import choice, seed, shuffle
from tensorflow import DType, Tensor, constant, float32, reshape
from tensorflow.python.data import Dataset
from wfdb import rdrecord

from mlbpestimation.data.datasetloader import DatasetLoader
from mlbpestimation.data.splitdataset import SplitDataset
from mlbpestimation.preprocessing.base import DatasetPreprocessingPipeline, PythonFunctionTransformOperation, TransformOperation


class MimicDatasetLoader(DatasetLoader):
    def __init__(self, mimic_wave_files_directory: str, random_seed: int, use_ppg: bool, subsample: float = 1.0):
        self.mimic_wave_files_directory = Path(mimic_wave_files_directory)
        self.random_seed = random_seed
        self.subsample = subsample
        self.input_signal = 'Pleth' if use_ppg else 'ABP'

    def load_datasets(self) -> SplitDataset:
        record_paths = self._get_paths()
        record_paths = self._shuffle_items(record_paths)
        record_paths = self._subsample_items(record_paths)
        record_paths_splits = self._make_splits(record_paths)

        datasets = []
        reading_pipeline = MimicReaderPreprocessingPipeline(self.input_signal)
        for path_split in record_paths_splits:
            dataset = Dataset.from_tensor_slices(path_split)
            dataset = reading_pipeline.apply(dataset)
            datasets.append(dataset)

        return SplitDataset(*datasets)

    def _get_paths(self):
        return [splitext(path)[0] for path in
                Path(self.mimic_wave_files_directory).rglob('*hea') if
                match(r'(\d)*.hea', path.name)]

    def _shuffle_items(self, record_paths):
        seed(self.random_seed)
        shuffle(record_paths)
        return record_paths

    def _subsample_items(self, record_paths):
        nb_records = len(record_paths)
        subsample_size = int(nb_records * self.subsample)
        return choice(record_paths, subsample_size, False)

    def _make_splits(self, record_paths):
        nb_records = len(record_paths)
        return split(record_paths, [int(nb_records * 0.70), int(nb_records * 0.85)])


class MimicReaderPreprocessingPipeline(DatasetPreprocessingPipeline):
    output_signal = 'ABP'

    def __init__(self, input_signal):
        super().__init__([
            ReadSignals((float32, float32), input_signal, self.output_signal),
            SetShape(),
        ])


class ReadSignals(PythonFunctionTransformOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], input_signal: str, output_signal):
        super().__init__(out_type)
        self.input_signal = input_signal
        self.output_signal = output_signal

    def transform(self, record_path: Tensor) -> Any:
        channel_names = {self.input_signal, self.output_signal}
        record = rdrecord(record_path.numpy().decode('ASCII'), channel_names=channel_names)
        signals = record.sig_name
        if signals is not None and self.input_signal in signals and self.output_signal in signals:
            input_index = signals.index(self.input_signal)
            output_index = signals.index(self.output_signal)
            return constant(record.p_signal[:, input_index], dtype=float32), constant(record.p_signal[:, output_index], dtype=float32)
        else:
            return constant(0, dtype=float32), constant(0, dtype=float32)


class SetShape(TransformOperation):
    def transform(self, input_signal: Tensor, output_signal: Tensor) -> Any:
        return reshape(input_signal, [-1]), reshape(output_signal, [-1])
