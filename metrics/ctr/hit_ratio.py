from typing import Union

from metrics.ctr.ctr_at_k_metric import CTRAtKMetric


class HitRatio(CTRAtKMetric):
    name = 'HR'
    group = True
    minimize = False

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        score_label_pairs = zip(scores, labels)
        sorted_score_label_pairs = sorted(score_label_pairs, key=lambda x: x[0], reverse=True)

        scores, labels = zip(*sorted_score_label_pairs)

        return int(1 in labels[:self.k])
