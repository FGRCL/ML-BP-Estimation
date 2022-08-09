from tensorflow import Tensor, constant, float32
from wfdb import rdrecord


class MimicCaseGenerator(object):
    def __init__(self, paths):
        self.paths = iter(paths)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self) -> Tensor:
        record = rdrecord(next(self.paths))
        while 'ABP' not in record.sig_name:
            record = rdrecord(next(self.paths))

        i = record.sig_name.index('ABP')
        return constant(record.p_signal[:, i], dtype=float32)
