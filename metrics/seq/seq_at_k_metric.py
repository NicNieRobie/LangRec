import abc

from metrics.seq.base_seq_metric import BaseSeqMetric
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class SeqAtKMetric(BaseSeqMetric, abc.ABC):
    k: int

    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def __str__(self):
        return f'{self.name}@{self.k}'
