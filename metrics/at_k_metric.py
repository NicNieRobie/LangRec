from metrics.base_metric import BaseMetric
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class AtKMetric(BaseMetric):
    k: int

    def __init__(self, k):
        self.k = k

    def __str__(self):
        return f'{self.name}@{self.k}'
