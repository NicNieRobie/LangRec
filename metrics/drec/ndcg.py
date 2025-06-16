import numpy as np

from metrics.drec.drec_at_k_metric import DrecAtKMetric


class NDCG(DrecAtKMetric):
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