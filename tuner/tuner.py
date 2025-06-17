import os.path
import os.path
import random
from typing import Type

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.discrete_code_preparer import DiscreteCodePreparer
from loader.preparer import Preparer
from model.base_model import BaseModel
from tuner.tune_utils.monitor import Monitor
from utils.timer import Timer
from loguru import logger


class Tuner:
    PREPARER_CLASS: Type[Preparer] = Preparer

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

        self.sign = f"{self.model_name}_on_{self.processor.dataset_name}_{self.config.task}"

        self.model_path = os.path.join(self.log_dir, f'{self.model_name}_on_{self.processor.dataset_name}.pt')
        self.log_path = os.path.join(self.log_dir, f'{self.model_name}_on_{self.processor.dataset_name}.log')

        self.model.prepare_model_finetuning(self.config, inference_mode=False, tune_from=self.config.tune_from)
        self.model.load()

        self.optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.model.model.parameters()),
            lr=self.config.lr
        )

        self.metrics_aggregator = self.build_metrics_aggregator()

        self.monitor = Monitor(metrics_aggregator=self.metrics_aggregator, patience=self.config.patience)
        self.latency_timer = Timer(activate=False)

    def build_metrics_aggregator(self):
        raise NotImplementedError

    def get_model(self):
        return self.model

    def load_model(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def evaluate(self, valid_dl, epoch):
        raise NotImplementedError

    def use_encoding(self):
        return self.config.code_path is not None or self.config.code_type is not None

    def get_eval_interval(self, total_train_steps):
        if self.config.eval_interval == 0:
            self.config.eval_interval = -1

        if self.config.eval_interval < 0:
            return total_train_steps // -self.config.eval_interval

        return self.config.eval_interval

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

        for index, batch in tqdm(enumerate(train_dl), total=total_train_steps, desc="Aligning"):
            if random.random() * self.config.align_step >= 1:
                continue

            loss = self.model.finetune(batch, alignment=True)
            loss.backward()

            accumulate_step += 1
            if accumulate_step == self.config.acc_batch:
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulate_step = 0

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
            for index, batch in tqdm(enumerate(train_dl), total=total_train_steps, desc="Tuning"):
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
                        logger.info('Early stopping')
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
        logger.info(f'Total {st.count} steps, avg ms {st.avgms():.4f}')

    def __call__(self):
        if self.config.latency:
            self.latency()
            return
        self.finetune()
