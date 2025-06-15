from typing import Union

from sklearn.metrics import label_ranking_average_precision_score

from metrics.ctr.base_ctr_metric import BaseCTRMetric


class MRR(BaseCTRMetric):
    name = 'MRR'
    group = True
    minimize = False

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        return label_ranking_average_precision_score([labels], [scores])
