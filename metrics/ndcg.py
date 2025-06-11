from typing import Union

from sklearn.metrics import ndcg_score

from metrics.at_k_metric import AtKMetric


class NDCG(AtKMetric):
    name = 'NDCG'
    group = True
    minimize = False

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        return ndcg_score([labels], [scores], k=self.k)
