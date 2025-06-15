from typing import Union

from sklearn.metrics import log_loss

from metrics.ctr.base_ctr_metric import BaseCTRMetric


class LogLoss(BaseCTRMetric):
    name = 'LogLoss'
    group = False
    minimize = True

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        return log_loss(labels, scores)
