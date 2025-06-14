import abc

from utils.load_processor import load_processor


class BaseEncoder(abc.ABC):
    DEFAULT_OUTPUT_DIR = 'encoding/'

    def __init__(self, config, device, output_dir=DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir

        self.config = config
        self.device = device

        self.dataset, self.task = config.dataset, config.task.lower()

        self.processor = load_processor(self.dataset, self.task)
        self.processor.load()

    def encode(self) -> dict:
        raise NotImplementedError
