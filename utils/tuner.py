import os.path
import os.path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.discrete_code_preparer import DiscreteCodePreparer
from loader.map import Map
from loader.preparer import Preparer
from model.base_model import BaseModel
from metrics.ctr.ctr_metrics_aggregator import CTRMetricsAggregator
from utils.dataloader import get_steps
from utils.discovery.class_library import ClassLibrary
from utils.gpu import get_device
from utils.monitor import Monitor
from utils.timer import Timer


class Tuner:
    PREPARER_CLASS = Preparer

    model: BaseModel

    def __init__(self, config, processor, model=None):
        self.config = config

        if model:
            self.model_name = model.get_name()
            self.model = model
        else:
            self.model_name = self.config.model.upper()
            self.model = self.load_model()

        self.processor = processor

        self.log_dir = os.path.join('tuning', self.model_name)
        os.makedirs(self.log_dir, exist_ok=True)

        print(f'python {" ".join(sys.argv)}')

        self.sign = f"{self.model_name}_on_{self.processor.dataset_name}_{self.config.task}"

        self.model_path = os.path.join(self.log_dir, f'{self.model_name}_on_{self.processor.dataset_name}.pt')
        self.log_path = os.path.join(self.log_dir, f'{self.model_name}_on_{self.processor.dataset_name}.log')

        self.model.prepare_model_finetuning(self.config, inference_mode=False, tune_from=self.config.tune_from)
        self.model.load()

        self.optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.model.model.parameters()),
            lr=self.config.lr
        )

        self.metrics_aggregator = CTRMetricsAggregator.build_from_config([self.config.valid_metric])

        self.monitor = Monitor(metrics_aggregator=self.metrics_aggregator, patience=self.config.patience)
        self.latency_timer = Timer(activate=False)

    def get_model(self):
        return self.model

    def load_model(self):
        models = ClassLibrary.models(self.config.task)

        if self.model_name not in models:
            raise ValueError(f'Unknown model: {self.model_name}')

        model = models[self.model_name]

        return model(device=get_device(self.config.gpu))

    @staticmethod
    def _get_steps(dataloader):
        return (len(dataloader.dataset) + dataloader.batch_size - 1) // dataloader.batch_size

    def evaluate(self, valid_dl, epoch):
        total_valid_steps = get_steps(valid_dl)

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

            results = self.metrics_aggregator(score_list, label_list, group_list)

            for k in results:
                metric_name = k
                metric_values.append(results[k])
            print(
                f'(epoch {epoch}) validation on {self.processor.dataset_name} dataset with {metric_name}: {metric_values[-1]:.4f}')
        self.model.model.train()

        metric_value = np.mean(metric_values).item()
        print(f'(epoch {epoch}) validation on all datasets with {metric_name}: {metric_value:.4f}')

        action = self.monitor.push(metric_name, metric_value)
        if action is self.monitor.BEST:
            self.model.save(os.path.join(self.log_dir, f'{self.sign}.pt'))
            print(f"Saving best model to {os.path.join(self.log_dir, f'{self.sign}.pt')}")
        return action

    def get_eval_interval(self, total_train_steps):
        if self.config.eval_interval == 0:
            self.config.eval_interval = -1

        if self.config.eval_interval < 0:
            return total_train_steps // -self.config.eval_interval

        return self.config.eval_interval

    def load_data(self):

        preparer = self.PREPARER_CLASS(
            processor=self.processor,
            model=self.model,
            config=self.config
        )
        train_df = preparer.load_or_generate(mode='train')

        preparer = self.PREPARER_CLASS(
            processor=self.processor,
            model=self.model,
            config=self.config
        )
        valid_dl = preparer.load_or_generate(mode='valid')

        return train_df, valid_dl

    def load_test_data(self):
        preparer = self.PREPARER_CLASS(
            processor=self.processor,
            model=self.model,
            config=self.config
        )
        if not preparer.has_generated:
            self.processor.load()
        test_dl = preparer.load_or_generate(mode='test')
        return test_dl

    def alignment(self):
        if not issubclass(self.PREPARER_CLASS, DiscreteCodePreparer):
            return

        if not self.config.alignment:
            return

        if hasattr(self, 'alignment_train_dl') and hasattr(self, 'alignment_total_train_steps'):
            train_dl = self.alignment_train_dl
            total_train_steps = self.alignment_total_train_steps
        else:
            preparer = self.PREPARER_CLASS(
                processor=self.processor,
                model=self.model,
                config=self.config
            )
            train_df = preparer.generate_item_alignment_data()
            train_ds = self.PREPARER_CLASS.DATASET_CLASS(train_df)
            train_dl = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
            self.__setattr__('alignment_train_dl', train_dl)
            total_train_steps = (len(train_ds) + self.config.batch_size - 1) // self.config.batch_size
            self.__setattr__('alignment_total_train_steps', total_train_steps)

        self.model.model.train()
        accumulate_step = 0
        self.optimizer.zero_grad()
        for index, batch in tqdm(enumerate(train_dl), total=total_train_steps):
            if random.random() * self.config.align_step >= 1:
                continue

            loss = self.model.finetune(batch, alignment=True)
            loss.backward()

            accumulate_step += 1
            if accumulate_step == self.config.acc_batch:
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulate_step = 0

    @property
    def test_command(self):
        return f'python main.py --model {self.model} --dataset <data_name>'

    def finetune(self):
        train_df, valid_dl = self.load_data()

        train_ds = self.PREPARER_CLASS.DATASET_CLASS(train_df)
        train_ds.align(batch_size=self.config.batch_size, ascending=False)
        train_dl = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=False)

        total_train_steps = (len(train_ds) + self.config.batch_size - 1) // self.config.batch_size

        eval_interval = self.get_eval_interval(total_train_steps)

        epoch = 0
        while epoch + 1 <= self.config.num_epochs:
            self.model.model.train()
            self.alignment()

            accumulate_step = 0
            self.optimizer.zero_grad()
            for index, batch in tqdm(enumerate(train_dl), total=total_train_steps):
                loss = self.model.finetune(batch)
                loss.backward()

                accumulate_step += 1
                if accumulate_step == self.config.acc_batch:
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
        if self.config.latency:
            self.latency()
            return
        self.finetune()
