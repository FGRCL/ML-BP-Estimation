from dataclasses import dataclass

import numpy as np


@dataclass
class FilteredAbpTrack:
    lowpass: np.ndarray
    bandpass: np.ndarray
