from typing import Union

import numpy as np
from numpy.core.multiarray import ndarray

from metrics.base_metric import BaseMetric


class BaseSeqMetric(BaseMetric):
    def __init__(self, prod_mode):
        self.prod_mode = prod_mode

    def _calculate(self, rank: int) -> Union[int, float]:
        raise NotImplementedError

    def calculate(self, ranks: list) -> ndarray:
        scores = [self._calculate(rank) for rank in ranks]

        if self.prod_mode:
            return np.prod(scores)

        return np.mean(scores)

    def __call__(self, *args, **kwargs) -> ndarray:
        return self.calculate(*args, **kwargs)
