import abc

from model.base_baseline_model import BaseBaselineModel

class BPRModel(BaseBaselineModel, abc.ABC):
    MODEL = 'BPR'

    def __init__(self, **kwargs):
        print(kwargs)
        super().__init__(**kwargs)
