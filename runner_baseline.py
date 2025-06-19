import os.path
import sys

import numpy as np
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data import data_preparation
from recbole.model.context_aware_recommender import DeepFM
from recbole.model.context_aware_recommender.autoint import AutoInt
from recbole.model.context_aware_recommender.dcn import DCN
from recbole.model.context_aware_recommender.dcnv2 import DCNV2
from recbole.model.context_aware_recommender.pnn import PNN
from recbole.model.general_recommender import BPR
from recbole.model.general_recommender.itemknn import ItemKNN
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer

from utils.discovery.class_library import ClassLibrary
from utils.export_writer import ExportWriter

class BaselineRunner:
    def __init__(self, config):
        self.config = config

        self.representation = config.representation

        self.task = config.task

        self.log_level = config.log_level

        self.model_name = config.model.upper()
        self.dataset = config.dataset.upper()

        self.dataset_file = None

        self.model = config.model

        self.sign = ''

        self.log_dir = os.path.join('export', self.model + '_' + self.dataset)

        os.makedirs(self.log_dir, exist_ok=True)

        self.exporter = ExportWriter(os.path.join(self.log_dir, f'{self.model_name}{self.sign}.dat'))

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

    def get_model(self):
        """translates model name to model instance - to simulate loading a model, like it was with LLMs"""
        if self.model == 'BPR':
            return BPR
        if self.model == 'DeepFM':
            return DeepFM
        if self.model == 'SASRec':
            return SASRec
        if self.model == 'AutoInt':
            return AutoInt
        if self.model == 'DCN':
            return DCN
        if self.model == 'DCNV2':
            return DCNV2
        if self.model == 'LightGDCN':
            return LightGCN
        if self.model == 'PNN':
            return PNN
        if self.model == 'BPR':
            return BPR
        if self.model == 'ItemKNN':
            return ItemKNN

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

        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=True)
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
            return ['user_id','item_id'], metrics, mode

    def set_representation(self):
        if self.representation == 'sem_id':
            return ['item_id', 'label_emb']
        return None

    def run(self):
        cols, metrics, mode = self.set_task()

        item_cols = self.set_representation()

        task = 'prediction'
        if self.task == 'seq' or self.task == 'drec': task = 'ranking'

        parameter_dict = {
            'dataset': self.dataset,
            'data_path': f'dataset_inter/{self.representation}/{self.task}/',
            'inter_file': f'{self.dataset}',
            'load_col': {'inter': cols, 'item': item_cols} if item_cols else {'inter': cols},
            'model': self.config.model,
            'epochs': self.config.epochs,
            'train_batch_size': 2048,
            'eval_batch_size': 4096,
            'eval_args': {
                'split': {'RS': [0.8, 0.1, 0.1]},
                'order': 'TO' if self.task == 'seq' else 'RO',
                'mode': mode
            },
            'metrics': metrics,
            'topk': self.config.topk,
            'task': task,
            'show_progress': True,
            'log_level': 'INFO' if self.log_level == 'true' else 'ERROR',
        }

        parameter_dict['item_file'] = f'{self.dataset}'

        if self.task == 'seq' or self.task == 'drec':
            parameter_dict['train_neg_sample_args'] = None


            parameter_dict['USER_ID_FIELD'] = 'user_id'
            parameter_dict['ITEM_ID_FIELD'] = 'item_id'

        train_data, valid_data, test_data, config, dataset = self.get_data(parameter_dict)

        best_valid_score, best_valid_result, trainer = self.run_model(train_data, valid_data, config, dataset)

        print(best_valid_score, best_valid_result)

        test_score = self.eval_model(trainer, test_data)

        print(test_score)