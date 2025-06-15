from metrics.seq.base_seq_metric import BaseSeqMetric


class MRR(BaseSeqMetric):
    name = 'MRR'
    group = True
    minimize = False

    def _calculate(self, rank: int):
        if rank < 0:
            return 0.0
        return 1.0 / rank
