from typing import Union

from metrics.at_k_metric import AtKMetric


class Recall(AtKMetric):
    name = 'Recall'
    group = True
    minimize = False

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        score_label_pairs = zip(scores, labels)
        sorted_score_label_pairs = sorted(score_label_pairs, key=lambda x: x[0], reverse=True)

        scores, labels = zip(*sorted_score_label_pairs)

        return sum(labels[:self.k]) * 1. / sum(labels)
