from metrics.ctr.ctr_metrics_aggregator import CTRMetricsAggregator
from metrics.seq.seq_metrics_aggregator import SeqMetricsAggregator


def get_metrics_aggregator(task: str, metrics_config, **kwargs):
    assert task in ["ctr", "seq", "drec"], "Unknown task"

    if task == "ctr":
        return CTRMetricsAggregator.build_from_config(metrics_config)
    elif task == "seq":
        return SeqMetricsAggregator.build_from_config(metrics_config, **kwargs)
    else:
        # TODO
        raise NotImplementedError
