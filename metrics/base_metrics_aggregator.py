import abc
from collections import OrderedDict
from typing import Type, Dict, List

from metrics.base_metric import BaseMetric


class BaseMetricsAggregator(abc.ABC):
    BASE_METRIC_CLS: Type[BaseMetric] = BaseMetric

    metric_dict: Dict[str, Type[BaseMetric]]

    def __init__(self, metrics: List[BASE_METRIC_CLS], metric_dict: dict):
        self.metrics = metrics
        self.metric_dict = metric_dict

        self.vals = OrderedDict()
        self.group = False

        for metric in self.metrics:
            self.vals[str(metric)] = []
            self.group = self.group or metric.group

    @classmethod
    def build_from_config(cls, **kwargs):
        raise NotImplementedError

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def is_minimize(self, metric: str):
        if isinstance(metric, self.BASE_METRIC_CLS):
            return metric.minimize

        assert isinstance(metric, str)

        metric = metric.split('@')[0]

        return self.metric_dict[metric].minimize
