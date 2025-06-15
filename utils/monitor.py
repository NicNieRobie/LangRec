class Monitor:
    BEST = 1
    SKIP = 2
    STOP = 3

    def __init__(self, metrics_aggregator, patience=2, warmup_steps=50, warmup_threshold=1e-6):
        self.metrics_aggregator = metrics_aggregator
        self.patience = patience
        self.best_value = None
        self.minimize = None
        self.best_index = 0
        self.current_index = -1

        self.warmup_steps = warmup_steps
        self.warmup_threshold = warmup_threshold
        self.in_warming_up = True

    def push(self, metric, value):
        self.current_index += 1

        if self.best_value is None:
            self.minimize = self.metrics_aggregator.is_minimize(metric)
            self.best_value = value
            return self.BEST

        if self.minimize ^ (value > self.best_value):
            self.best_value = value
            self.best_index = self.current_index
            if value > self.warmup_threshold:
                self.in_warming_up = False
            return self.BEST

        if (self.in_warming_up and
                not self.minimize and
                self.best_value < self.warmup_threshold and
                self.current_index < self.warmup_steps):
            return self.SKIP

        if self.current_index - self.best_index >= self.patience:
            return self.STOP
        return self.SKIP