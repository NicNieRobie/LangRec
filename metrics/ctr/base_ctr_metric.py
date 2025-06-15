from typing import Union

from metrics.base_metric import BaseMetric


class BaseCTRMetric(BaseMetric):
    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        pass

    def __call__(self, *args, **kwargs) -> Union[int, float]:
        return self.calculate(*args, **kwargs)
