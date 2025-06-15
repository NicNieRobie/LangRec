from typing import Union

from sklearn.metrics import roc_auc_score

from metrics.ctr.base_ctr_metric import BaseCTRMetric


class AUC(BaseCTRMetric):
    name = 'AUC'
    group = False
    minimize = False

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        return roc_auc_score(labels, scores)
