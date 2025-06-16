from metrics.drec.base_drec_metric import BaseDrecMetric


class MRR(BaseDrecMetric):
    name = 'MRR'
    group = True
    minimize = False

    def _calculate(self, rank: int):
        if rank < 0:
            return 0.0
        return 1.0 / rank
