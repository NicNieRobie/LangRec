import numpy as np

from metrics.seq.seq_at_k_metric import SeqAtKMetric


class NDCG(SeqAtKMetric):
    name = 'NDCG'
    group = True
    minimize = False

    def _calculate(self, rank: int):
        if self.prod_mode:
            if rank < 1:
                return 0
            return 1.0 / np.log2(rank + 1)

        if rank < 1 or rank > self.n:
            return 0

        return 1.0 / np.log2(rank + 1)