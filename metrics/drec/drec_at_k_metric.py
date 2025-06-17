import abc

from metrics.drec.base_drec_metric import BaseDrecMetric
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class DrecAtKMetric(BaseDrecMetric, abc.ABC):
    k: int

    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def __str__(self):
        return f'{self.name}@{self.k}'
