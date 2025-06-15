from metrics.ctr.base_ctr_metric import BaseCTRMetric
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class CTRAtKMetric(BaseCTRMetric):
    k: int

    def __init__(self, k):
        self.k = k

    def __str__(self):
        return f'{self.name}@{self.k}'
