class BaseMetric:
    name: str
    group: bool
    minimize: bool

    def __str__(self):
        return self.name
