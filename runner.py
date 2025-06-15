from data.base_processor import BaseProcessor
from model.seq.base_seq_model import BaseSeqModel
from utils.code import get_code_indices
from utils.discovery.class_library import ClassLibrary
from utils.gpu import GPU
from utils.seq_tuner import SeqTuner
from utils.tester import Tester
from utils.tuner import Tuner


class Runner:
    def __init__(self, config):
        self.config = config

        assert config.task in ["ctr", "drec", "seq"]
        self.task = config.task

        assert config.mode in ["test", "finetune", "testtune"]
        self.subset = config.mode if config.mode in ["test", "finetune"] else "original"

        self.model_name = config.model.upper()
        self.dataset = config.dataset.upper()

        self.processor: BaseProcessor = self.load_processor()
        self.processor.load()

        if self.config.mode in ["finetune", "testtune"]:
            if config.task == 'seq':
                self.tuner = SeqTuner(self.config, self.processor)
            else:
                self.tuner = Tuner(self.config, self.processor)

            self.model = self.tuner.get_model()
        else:
            self.model = self.load_model()

        if self.config.mode in ["test", "testtune"]:
            self.tester = Tester(self.config, self.processor, self.model)

    def load_processor(self, data_path=None):
        processors = ClassLibrary.processors(self.task)

        if self.dataset not in processors:
            raise ValueError(f'Unknown dataset: {self.dataset}')

        processor = processors[self.dataset]

        return processor(data_path=data_path)

    def load_model(self):
        models = ClassLibrary.models(self.task)

        if self.model_name not in models:
            raise ValueError(f'Unknown model: {self.model_name}')

        model = models[self.model_name]

        if issubclass(model, BaseSeqModel):
            _, code_list, num_codes = get_code_indices(self.config.code_path)

            return model(device=self.get_device(), num_codes=num_codes, code_list=code_list).load()
        else:
            return model(device=self.get_device()).load()

    def get_device(self):
        if self.config.gpu is None:
            return GPU.auto_choose(torch_format=True)

        if self.config.gpu == -1:
            print('Choosing CPU device')
            return 'cpu'

        print(f'Choosing {self.config.gpu}-th GPU')
        return f'CUDA: {self.config.gpu}'

    def run(self):
        if self.config.mode in ["finetune", "testtune"]:
            self.tuner()

        if self.config.mode in ["test", "testtune"]:
            self.tester()
