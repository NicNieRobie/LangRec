from multiprocessing import Pool
from typing import List

import pandas as pd
import torch

from metrics.base_metrics_aggregator import BaseMetricsAggregator
from metrics.ctr.base_ctr_metric import BaseCTRMetric
from utils.discovery.class_library import ClassLibrary


class CTRMetricsAggregator(BaseMetricsAggregator):
    BASE_METRIC_CLS = BaseCTRMetric

    def __init__(self, metrics: List[BASE_METRIC_CLS], metric_dict: dict):
        super().__init__(metrics, metric_dict)

    @classmethod
    def build_from_config(cls, metrics_config):
        metrics = ClassLibrary.ctr_metrics()

        metric_dict = {m.name.upper(): m for name, m in metrics.class_dict.items()}

        metrics = []

        for m_str in metrics_config:
            at_idx = m_str.find('@')
            arguments = []

            if at_idx > -1:
                m_str, arguments = m_str[:at_idx], [int(m_str[at_idx + 1:])]

            if m_str.upper() not in metric_dict:
                raise ValueError(f'Metric {m_str} not found')

            metrics.append(metric_dict[m_str.upper()](*arguments))

        return cls(metrics, metric_dict)

    def evaluate(self, scores, labels, groups, group_worker=5):
        if not self.metrics:
            return {}

        df = pd.DataFrame(dict(groups=groups, scores=scores, labels=labels))

        groups = None
        if self.group:
            groups = df.groupby('groups')

        for metric in self.metrics:
            if not metric.group:
                self.vals[str(metric)] = metric(scores=scores, labels=labels)
                continue

            tasks = []
            pool = Pool(processes=group_worker)

            for g in groups:
                group = g[1]

                g_labels = group.labels.tolist()
                g_scores = group.scores.tolist()

                tasks.append(pool.apply_async(metric, args=(g_scores, g_labels)))

            pool.close()
            pool.join()

            vals = [t.get() for t in tasks]

            self.vals[str(metric)] = torch.tensor(vals, dtype=torch.float).mean().item()

        return self.vals

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def is_minimize(self, metric: str):
        if isinstance(metric, BaseCTRMetric):
            return metric.minimize

        assert isinstance(metric, str)

        metric = metric.split('@')[0]

        return self.metric_dict[metric].minimize
