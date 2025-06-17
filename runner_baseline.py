import os.path
import sys

import numpy as np
import torch
from recbole.model.context_aware_recommender import DeepFM
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

from metrics.base_metrics_aggregator import BaseMetricsAggregator
from recbole.data import data_preparation
from recbole.model.sequential_recommender import SASRec
import pandas as pd


class BaselineRunner:
    def __init__(self, config):
        self.config = config

        self.task = config.task

        self.model_name = config.model.upper()
        self.dataset = config.dataset.upper()

        # self.processor: BaseProcessor = self.load_processor()
        # self.processor.load()

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
        processors = ClassLibrary.processors(self.task)

        if self.dataset not in processors:
            raise ValueError(f'Unknown dataset: {self.dataset}')

        processor = processors[self.dataset]

        return processor(data_path=data_path)

    def load_model(self):
        models = ClassLibrary.baseline_models(self.task)

        if self.model_name not in models:
            raise ValueError(f'Unknown model: {self.model_name}')

        model = models[self.model_name]

        return model

    def get_device(self):
        return 'cpu'
        # if self.config.gpu is None:
        #     return GPU.auto_choose()
        #
        # if self.config.gpu == -1:
        #     print('Choosing CPU device')
        #     return 'cpu'
        #
        # print(f'Choosing {self.config.gpu}-th GPU')
        # return f'CUDA: {self.config.gpu}'

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

        aggregator = BaseMetricsAggregator.build_from_config(self.config.metrics)

        results = aggregator(scores, labels, groups)

        for metric, val in results.items():
            print(f'{metric}: {val:.4f}')

        self.exporter.save_metrics(results)

    # def get_dataset(self):
    #     """creates inter file"""
    #
    #     if self.task == 'ctr':
    #         parquet_path = f'data_store/{self.dataset.lower()}/interactions.parquet'
    #
    #         df = pd.read_parquet(parquet_path)
    #         df.rename(columns={df.columns[0]: 'user_id:token', df.columns[1]: 'item_id:token', df.columns[2]: f'label:float'}, inplace=True)
    #         df.to_csv(f'dataset_inter/{self.task}/{self.dataset}/{self.dataset}.inter', sep='\t', index=False)
    #
    #     if self.task == 'seq' or self.task == 'drec':
    #         if os.path.exists(f'dataset_inter/{self.task}/{self.dataset}/{self.dataset}.inter'):
    #

        # parquet_path = f'data_store/{self.dataset.lower()}/users.parquet'
            #
            # df = pd.read_parquet(parquet_path)
            #
            # records = []
            # for _, row in df.iterrows():
            #     uid = row['uid']
            #     items = row['history']
            #     for i, item_id in enumerate(items):
            #         records.append([uid, item_id, i + 1])
            #
            # inter_df = pd.DataFrame(records, columns=['user_id:token', 'item_id:token', 'timestamp:float'])
            #
            # inter_df.to_csv(f'dataset_inter/{self.task}/{self.dataset}/{self.dataset}.inter', sep='\t', index=False)

    def get_model(self):
        """translates model name to model instance - to simulate loading a model, like it was with LLMs"""
        if self.model == 'BPR':
            return BPR
        if self.model == 'DeepFM':
            return DeepFM
        if self.model == 'SASRec':
            return SASRec
        return ValueError('Unknown model')

    def get_data(self, parameter_dict):
        parameter_dict['topk'] = [int(k) for k in parameter_dict['topk']]

        sys.argv = [sys.argv[0]]  # clean cmd arguments so that config actually takes parameter_dict and not sys.argv

        config = Config(model=self.model, config_dict=parameter_dict) # create recbole config

        dataset = create_dataset(config) # create recbole dataset

        train_data, valid_data, test_data = data_preparation(config, dataset) # split data
        return train_data, valid_data, test_data, config, dataset

    def run_model(self, train_data, valid_data, config, dataset):
        model = self.get_model()
        model = model(config, dataset).to(config['device'])

        trainer = Trainer(config, model) # load model instance and config into trainer

        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
        return best_valid_score, best_valid_result, trainer

    def eval_model(self, trainer, test_data):
        test_result = trainer.evaluate(test_data)
        return test_result

    def set_task(self):
        if self.task == 'ctr':
            metrics = ['AUC']
            mode = 'labeled'
            return ['user_id', 'item_id', 'label'], metrics, mode
        if self.task == 'seq':
            metrics = ['Recall', 'MRR', 'NDCG']
            mode = 'full'
            return ['user_id','item_id', 'timestamp'], metrics, mode
        if self.task == 'drec':
            metrics = ['Recall', 'MRR', 'NDCG']
            mode = 'full'
            return ['user_id','candidates', 'label'], metrics, mode

    def run(self):
        # self.get_dataset()

        cols, metrics, mode = self.set_task()

        task = 'prediction'
        if self.task == 'seq' or self.task == 'drec': task = 'ranking'

        parameter_dict = {
            'dataset': self.dataset,
            'data_path': f'dataset_inter/{self.task}',
            'inter_file': self.dataset,
            'load_col': {'inter': cols},
            'model': self.config.model,
            'epochs': self.config.epochs,
            'train_batch_size': 2048,
            'eval_batch_size': 4096,
            'eval_args': {
                'split': {'RS': [0.8, 0.1, 0.1]},
                'order': 'TO' if self.task == 'seq' or self.task == 'drec' else 'RO',
                'mode': mode
            },
            'metrics': metrics,
            'topk': self.config.topk,
            'task': task,
            'show_progress': True,
        }

        if self.task == 'seq':
            parameter_dict['train_neg_sample_args'] = None

        # if self.task == 'drec':
        #     parameter_dict['dataset_class'] = 'CandidateRankingDataset'

        train_data, valid_data, test_data, config, dataset = self.get_data(parameter_dict)

        best_valid_score, best_valid_result, trainer = self.run_model(train_data, valid_data, config, dataset)

        print(best_valid_score, best_valid_result)

        test_score = self.eval_model(trainer, test_data)

        print(test_score)