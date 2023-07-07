from typing import Any

from mlbpestimation.configuration.decorators import configuration


@configuration('base_dataset_configuration', 'hypothesis/dataset')
class DatasetConfiguration:
    source: Any
    decorators: Any
