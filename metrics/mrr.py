from typing import Union

from sklearn.metrics import label_ranking_average_precision_score

from metrics.base_metric import BaseMetric


class MRR(BaseMetric):
    name = 'MRR'
    group = True
    minimize = False

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        return label_ranking_average_precision_score([labels], [scores])
