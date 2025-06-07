from typing import List
from collections import OrderedDict
from multiprocessing import Pool

import pandas as pd
import torch

from utils.discovery.class_library import ClassLibrary
from metrics.base_metric import BaseMetric


class MetricsAggregator:
    def __init__(self, metrics: List[BaseMetric], metric_dict: dict):
        self.metrics = metrics
        self.metric_dict = metric_dict

        self.vals = OrderedDict()
        self.group = False

        for metric in self.metrics:
            self.vals[str(metric)] = []
            self.group = self.group or metric.group

    @classmethod
    def build_from_config(cls, metrics_config):
        metrics = ClassLibrary.metrics()

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
        if isinstance(metric, BaseMetric):
            return metric.minimize

        assert isinstance(metric, str)

        metric = metric.split('@')[0]

        return self.metric_dict[metric].minimize
