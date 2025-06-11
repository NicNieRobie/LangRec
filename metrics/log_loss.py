from typing import Union

from sklearn.metrics import log_loss

from metrics.base_metric import BaseMetric


class LogLoss(BaseMetric):
    name = 'LogLoss'
    group = False
    minimize = True

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        return log_loss(labels, scores)
