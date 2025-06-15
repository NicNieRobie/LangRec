from multiprocessing import Pool
from typing import List

import pandas as pd
import torch

from metrics.base_metrics_aggregator import BaseMetricsAggregator
from metrics.seq.base_seq_metric import BaseSeqMetric
from utils.discovery.class_library import ClassLibrary


class SeqMetricsAggregator(BaseMetricsAggregator):
    BASE_METRIC_CLS = BaseSeqMetric

    def __init__(self, metrics: List[BASE_METRIC_CLS], metric_dict: dict):
        super().__init__(metrics, metric_dict)

    @classmethod
    def build_from_config(cls, metrics_config, num_items, prod_mode):
        metrics = ClassLibrary.seq_metrics()

        metric_dict = {m.name.upper(): m for name, m in metrics.class_dict.items()}

        metrics = []

        for m_str in metrics_config:
            at_idx = m_str.find('@')
            arguments = []

            if at_idx > -1:
                m_str, arguments = m_str[:at_idx], [int(m_str[at_idx + 1:])]

            if m_str.upper() not in metric_dict:
                raise ValueError(f'Metric {m_str} not found')

            metrics.append(metric_dict[m_str.upper()](num_items=num_items, prod_mode=prod_mode, *arguments))

        return cls(metrics, metric_dict)

    def evaluate(self, ranks, groups, group_worker=5):
        if not self.metrics:
            return {}

        df = pd.DataFrame(dict(groups=groups, ranks=ranks))

        groups = None
        if self.group:
            groups = df.groupby('groups')

        for metric in self.metrics:
            if not metric.group:
                self.vals[str(metric)] = metric(scores=ranks)
                continue

            tasks = []
            pool = Pool(processes=group_worker)

            for g in groups:
                group = g[1]

                g_ranks = group.ranks.tolist()

                tasks.append(pool.apply_async(metric, args=(g_ranks,)))

            pool.close()
            pool.join()

            vals = [t.get() for t in tasks]

            self.vals[str(metric)] = torch.tensor(vals, dtype=torch.float).mean().item()

        return self.vals

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def is_minimize(self, metric: str):
        if isinstance(metric, BaseSeqMetric):
            return metric.minimize

        assert isinstance(metric, str)

        metric = metric.split('@')[0]

        return self.metric_dict[metric].minimize
