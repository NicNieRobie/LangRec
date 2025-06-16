from metrics.drec.drec_at_k_metric import DrecAtKMetric


class HitRatio(DrecAtKMetric):
    name = 'HR'
    group = True
    minimize = False

    def _calculate(self, rank: int):
        if rank < 0:
            return 0
        return int(rank <= self.k)
