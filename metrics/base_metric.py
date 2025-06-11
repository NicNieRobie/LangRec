from typing import Union


class BaseMetric:
    name: str
    group: bool
    minimize: bool

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        pass

    def __call__(self, *args, **kwargs) -> Union[int, float]:
        return self.calculate(*args, **kwargs)

    def __str__(self):
        return self.name
