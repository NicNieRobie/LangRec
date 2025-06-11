from typing import Union

from sklearn.metrics import roc_auc_score

from metrics.base_metric import BaseMetric


class GAUC(BaseMetric):
    name = 'GAUC'
    group = True
    minimize = False

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        return roc_auc_score(labels, scores)
