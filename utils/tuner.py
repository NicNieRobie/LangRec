import json
import os.path
import random
import sys

import numpy as np
import torch
from box import Box
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.discovery.class_library import ClassLibrary
from loader.discrete_code_preparer import DiscreteCodePreparer
from loader.map import Map
from loader.preparer import Preparer
from data.base_processor import BaseProcessor
from utils.gpu import GPU
from utils.metrics import MetricPool
from utils.monitor import Monitor
from utils.timer import Timer


class Tuner:
    PREPARER_CLASS = Preparer

    def __init__(self, conf, processor, model):
        self.conf = conf
        self.model_name = model.get_name()
        self.model = model

        self.processor = processor # type: BaseProcessor

        self.log_dir = os.path.join('tuning', self.model_name)
        os.makedirs(self.log_dir, exist_ok=True)

        print(f'python {" ".join(sys.argv)}')

        self.sign = f"{self.model_name}_on_{self.processor.dataset_name}"

        self.model_path = os.path.join(self.log_dir, f'{self.model_name}_on_{self.processor.dataset_name}.pt')
        self.log_path = os.path.join(self.log_dir, f'{self.model_name}_on_{self.processor.dataset_name}.log')

        self.model.prepare_model_finetuning(self.conf, inference_mode=False, tune_from=self.conf.tune_from)
        self.model.post_init()

        self.optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.model.model.parameters()),
            lr=self.conf.lr
        )

        self.monitor = Monitor(patience=self.conf.patience)
        self.latency_timer = Timer(activate=False)

    def get_device(self):
        if self.conf.gpu is None:
            return GPU.auto_choose(torch_format=True)
        if self.conf.gpu == -1:
            print('manually choosing CPU device')
            return 'cpu'

        print(f'manually choosing {self.conf.gpu}-th GPU')
        return f'cuda:{self.conf.gpu}'

    def load_model(self):
        models = ClassLibrary.models(self.conf.task)
        if self.model in models:
            model = models[self.model]
            print(f'loading {model.get_name()} model')
            return model(device=self.get_device())
        raise ValueError(f'unknown model: {self.model}')

    @staticmethod
    def _get_steps(dataloader):
        return (len(dataloader.dataset) + dataloader.batch_size - 1) // dataloader.batch_size

    def list_tunable_parameters(self):
        print('tunable parameters:')
        for name, param in self.model.model.named_parameters():
            if param.requires_grad:
                print(f'{name}: {param.size()}')

    def evaluate(self, valid_dl, epoch):
        total_valid_steps = self._get_steps(valid_dl)

        self.model.model.eval()
        with torch.no_grad():
            metric_name, metric_values = None, []
            print(f'(epoch {epoch}) validating: {self.processor.dataset_name}')
            score_list, label_list, group_list = [], [], []
            for index, batch in enumerate(tqdm(valid_dl, total=total_valid_steps)):
                self.latency_timer.run('test')
                scores = self.model.evaluate(batch)
                self.latency_timer.run('test')
                labels = batch[Map.LBL_COL].tolist()
                groups = batch[Map.UID_COL].tolist()

                score_list.extend(scores)
                label_list.extend(labels)
                group_list.extend(groups)

            pool = MetricPool.parse([self.conf.valid_metric])
            results = pool.calculate(score_list, label_list, group_list)
            for k in results:
                metric_name = k
                metric_values.append(results[k])
            print(f'(epoch {epoch}) validation on {self.processor.dataset_name} dataset with {metric_name}: {metric_values[-1]:.4f}')
        self.model.model.train()

        metric_value = np.mean(metric_values).item()
        print(f'(epoch {epoch}) validation on all datasets with {metric_name}: {metric_value:.4f}')

        action = self.monitor.push(metric_name, metric_value)
        if action is self.monitor.BEST:
            self.model.save(os.path.join(self.log_dir, f'{self.sign}.pt'))
            print(f'saving best model to {self.log_dir}/{self.sign}.pt')
        return action

    def get_eval_interval(self, total_train_steps):
        if self.conf.eval_interval == 0:
            self.conf.eval_interval = -1

        if self.conf.eval_interval < 0:
            return total_train_steps // -self.conf.eval_interval

        return self.conf.eval_interval

    def load_data(self):

        preparer = self.PREPARER_CLASS(
            processor=self.processor,
            model=self.model,
            conf=self.conf
        )
        train_df = preparer.load_or_generate(mode='train')

        preparer = self.PREPARER_CLASS(
            processor=self.processor,
            model=self.model,
            conf=self.conf
        )
        valid_dl = preparer.load_or_generate(mode='valid')

        return train_df, valid_dl

    def load_test_data(self):
        preparer = self.PREPARER_CLASS(
            processor=self.processor,
            model=self.model,
            conf=self.conf
        )
        if not preparer.has_generated:
            self.processor.load()
        test_dl = preparer.load_or_generate(mode='test')
        return test_dl

    def alignment(self):
        if not issubclass(self.PREPARER_CLASS, DiscreteCodePreparer):
            return

        if not self.conf.alignment:
            return

        if hasattr(self, 'alignment_train_dl') and hasattr(self, 'alignment_total_train_steps'):
            train_dl = self.alignment_train_dl
            total_train_steps = self.alignment_total_train_steps
        else:
            preparer = self.PREPARER_CLASS(
                processor=self.processor,
                model=self.model,
                conf=self.conf
            )
            train_df = preparer.generate_item_alignment_data()
            train_ds = self.PREPARER_CLASS.DATASET_CLASS(train_df)
            train_dl = DataLoader(train_ds, batch_size=self.conf.batch_size, shuffle=True)
            self.__setattr__('alignment_train_dl', train_dl)
            total_train_steps = (len(train_ds) + self.conf.batch_size - 1) // self.conf.batch_size
            self.__setattr__('alignment_total_train_steps', total_train_steps)

        self.model.model.train()
        accumulate_step = 0
        self.optimizer.zero_grad()
        for index, batch in tqdm(enumerate(train_dl), total=total_train_steps):
            if random.random() * self.conf.align_step >= 1:
                continue

            loss = self.model.finetune(batch, alignment=True)
            loss.backward()

            accumulate_step += 1
            if accumulate_step == self.conf.acc_batch:
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulate_step = 0

    @property
    def test_command(self):
        return f'python main.py --model {self.model} --dataset <data_name>'

    def finetune(self):
        train_df, valid_dl = self.load_data()

        train_ds = self.PREPARER_CLASS.DATASET_CLASS(train_df)
        train_ds.align(batch_size=self.conf.batch_size, ascending=False)
        train_dl = DataLoader(train_ds, batch_size=self.conf.batch_size, shuffle=False)

        total_train_steps = (len(train_ds) + self.conf.batch_size - 1) // self.conf.batch_size

        self.list_tunable_parameters()

        eval_interval = self.get_eval_interval(total_train_steps)

        epoch = 0
        while epoch + 1 <= self.conf.num_epochs:
            self.model.model.train()
            self.alignment()

            accumulate_step = 0
            self.optimizer.zero_grad()
            for index, batch in tqdm(enumerate(train_dl), total=total_train_steps):
                loss = self.model.finetune(batch)
                loss.backward()

                accumulate_step += 1
                if accumulate_step == self.conf.acc_batch:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulate_step = 0

                if (index + 1) % eval_interval == 0:
                    action = self.evaluate(valid_dl, epoch)
                    if action is self.monitor.STOP:
                        print('early stopping')
                        print(f'please evaluate the model by: {self.test_command}')
                        return

            epoch += 1

    def latency(self):
        self.latency_timer.activate()
        self.latency_timer.clear()
        train_dfs, valid_dls = self.load_data()

        try:
            self.evaluate(valid_dls, 0)
        except KeyboardInterrupt:
            pass
        st = self.latency_timer.status_dict['test']
        print(f'Total {st.count} steps, avg ms {st.avgms():.4f}')

    def __call__(self):
        if self.conf.latency:
            self.latency()
            return
        self.finetune()