from typing import Union

from sklearn.metrics import ndcg_score

from metrics.ctr.ctr_at_k_metric import CTRAtKMetric


class NDCG(CTRAtKMetric):
    name = 'NDCG'
    group = True
    minimize = False

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        return ndcg_score([labels], [scores], k=self.k)
