import os.path
import sys

import numpy as np
import torch
from recbole.quick_start import run_recbole
from setuptools.command.setopt import config_file
from recbole.config import Config
from recbole.data import create_dataset
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from data.base_processor import BaseProcessor
from utils import bars
from utils.discovery.class_library import ClassLibrary
from utils.exporter import Exporter
from utils.gpu import GPU

from metrics.metrics_aggregator import MetricsAggregator
from recbole.data import data_preparation


class BaselineRunner:
    def __init__(self, config):
        self.config = config

        self.model_name = config.model.upper()
        self.dataset = config.dataset.upper()

        self.processor: BaseProcessor = self.load_processor()
        self.processor.load()

        self.dataset_file = None

        self.model = config.model
        #self.load_model()

        self.sign = ''

        self.log_dir = os.path.join('export', self.model + '_' + self.dataset)

        os.makedirs(self.log_dir, exist_ok=True)

        self.exporter = Exporter(os.path.join(self.log_dir, f'{self.model_name}{self.sign}.dat'))

        if self.config.rerun:
            self.exporter.reset()

    def load_processor(self, data_path=None):
        processors = ClassLibrary.processors()

        if self.dataset not in processors:
            raise ValueError(f'Unknown dataset: {self.dataset}')

        processor = processors[self.dataset]

        return processor(data_path=data_path)

    def load_model(self):
        models = ClassLibrary.baseline_models()

        if self.model_name not in models:
            raise ValueError(f'Unknown model: {self.model_name}')

        model = models[self.model_name]

        return model

    def get_device(self):
        return 'cpu'
        if self.config.gpu is None:
            return GPU.auto_choose()

        if self.config.gpu == -1:
            print('Choosing CPU device')
            return 'cpu'

        print(f'Choosing {self.config.gpu}-th GPU')
        return f'CUDA: {self.config.gpu}'

    @staticmethod
    def _truncate_inputs(history, candidate, top_k_ratio=0.1):
        lengths = [len(item) for item in history]
        sorted_indices = np.argsort(lengths)[::-1].tolist()
        top_k = max(int(len(sorted_indices) * top_k_ratio), 1)

        for i in sorted_indices[:top_k]:
            history[i] = history[i][:max(len(history[i]) // 2, 10)]

        return history, candidate[:max(len(candidate) // 2, 10)]

    def _retry_with_truncation(self, history, candidate, input_template):
        for _ in range(5):
            for i in range(len(history)):
                _history = [f'({j + 1}) {history[i + j]}' for j in range(len(history) - i)]
                input_sequence = input_template.format('\n'.join(_history), candidate)
                response = self.model(input_sequence)

                if response is not None:
                    return response

            history, candidate = self._truncate_inputs(history, candidate)

        return None

    def evaluate(self):
        scores = self.exporter.read()

        source_set = self.processor.get_source_set(self.config.source)

        labels = source_set[self.processor.LABEL_COL].values
        groups = source_set[self.processor.USER_ID_COL].values

        aggregator = MetricsAggregator.build_from_config(self.config.metrics)

        results = aggregator(scores, labels, groups)

        for metric, val in results.items():
            print(f'{metric}: {val:.4f}')

        self.exporter.save_metrics(results)

    def get_dataset(self):
        if self.dataset == 'MOVIELENS': self.dataset_file = 'dataset'
        else: return ValueError(f"No dataset named {self.dataset}")

    def get_model(self):
        if self.model == 'BPR':
            return BPR

    def get_data(self, parameter_dict):
        parameter_dict['topk'] = [int(k) for k in parameter_dict['topk']]

        sys.argv = [sys.argv[0]]  # clean cmd arguments so that config actually takes parameter_dict and not sys.argv

        # Create config
        config = Config(model=self.model, config_dict=parameter_dict, config_file_list=[])

        # Load and process dataset
        dataset = create_dataset(config)

        train_data, valid_data, test_data = data_preparation(config, dataset)
        return train_data, valid_data, test_data, config, dataset

    def run_model(self, train_data, valid_data, config, dataset):
        model = self.get_model()
        model = model(config, dataset).to(config['device'])

        # Initialize trainer and start training
        trainer = Trainer(config, model)

        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
        return best_valid_score, best_valid_result, trainer

    def eval_model(self, trainer, test_data):
        test_result = trainer.evaluate(test_data)
        return test_result

    def run(self):
        self.get_dataset()

        print('metrics:', self.config.metrics, type(self.config.metrics))

        parameter_dict = {
            'dataset': self.dataset,
            'data_path': self.dataset_file,
            'inter_file': 'MOVIELENS.inter',
            'load_col': {
                'inter': ['user_id', 'item_id', 'click']
            },
            'model': self.config.model,
            'epochs': self.config.epochs,
            'train_batch_size': 2048,
            'eval_batch_size': 4096,
            'eval_args': {
                'split': {'RS': [0.8, 0.1, 0.1]},
                'order': 'RO',
                'mode': 'full'
            },
            'metrics': self.config.metrics,
            'topk': self.config.topk
        }

        train_data, valid_data, test_data, config, dataset = self.get_data(parameter_dict)

        best_valid_score, best_valid_result, trainer = self.run_model(train_data, valid_data, config, dataset)

        print(best_valid_score, best_valid_result)

        test_score = self.eval_model(trainer, test_data)

        print(test_score)
