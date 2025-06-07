from metrics.base_metric import BaseMetric
from utils.class_library import ignore_discovery


@ignore_discovery
class AtKMetric(BaseMetric):
    k: int

    def __init__(self, k):
        self.k = k

    def __str__(self):
        return f'{self.name}@{self.k}'
