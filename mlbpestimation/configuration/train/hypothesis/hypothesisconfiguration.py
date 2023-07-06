from typing import Any

from omegaconf import MISSING

from mlbpestimation.configuration.decorators import configuration
from mlbpestimation.configuration.train.hypothesis.dataset.datasetconfiguration import DatasetConfiguration


@configuration('basehypothesis', 'hypothesis')
class HypothesisConfiguration:
    dataset: DatasetConfiguration = MISSING
    model: Any = MISSING
    optimization: Any = MISSING
    output_directory: str = '${directories.output}'
