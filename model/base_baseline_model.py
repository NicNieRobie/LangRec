from recbole.quick_start import run_recbole

class BaseBaselineModel:
    MODEL = None

    def __init__(self, device, config_file, dataset_file):
        self.device = device
        self.config_file = config_file
        self.dataset_file = dataset_file

    @classmethod
    def get_name(cls):
        return cls.MODEL