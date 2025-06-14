import os

from transformers import Trainer, TrainingArguments

from utils import bars
from utils.exporter import Exporter

class Tuner:
    def __init__(self, config, processor, model):
        raise NotImplementedError()