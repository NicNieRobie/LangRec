import os
import random
from typing import cast

import torch
from tqdm import tqdm

from loader.map import Map
from loader.drec_preparer import DrecPreparer
from metrics.drec.drec_metrics_aggregator import DrecMetricsAggregator
from model.drec.base_drec_model import BaseDrecModel
from utils.code import get_code_indices
from utils.dataloader import get_steps
from utils.discovery.class_library import ClassLibrary
from utils.gpu import get_device
from tuner.tuner import Tuner


class DrecTuner(Tuner):
    PREPARER_CLASS = DrecPreparer

    model: BaseDrecModel
    num_codes: int

    def build_metrics_aggregator(self):
        return DrecMetricsAggregator.build_from_config(
            [self.config.valid_metric],
            prod_mode=self.config.search_mode == 'prod'
        )

    def load_model(self):
        device = get_device(self.config.gpu)
        _, code_list, self.num_codes = get_code_indices(self.config, device)

        models = ClassLibrary.models(self.config.task)

        if self.model_name not in models:
            raise ValueError(f'Unknown model: {self.model_name}')

        model = models[self.model_name]

        assert issubclass(model, BaseDrecModel), f'{model} is not a subclass of BaseDrecModel'

        return model(device=device, num_codes=self.num_codes, code_list=code_list, task=self.config.task)

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

        cast(BaseDrecModel, self.model).set_code_meta(preparer.code_tree, preparer.code_map)

        return train_df, valid_dl

    def _evaluate(self, dataloader, steps, step=1):
        search_mode = self.config.search_mode

        assert search_mode in ['prod', 'list',
                               'tree'], f'Unknown search mode {search_mode}, should be one of: prod, list, tree'

        group_list, ranks_list = [], []
        item_index = 0
        for index, batch in tqdm(enumerate(dataloader), total=steps, desc="Validating"):
            if random.random() * step > 1:
                continue

            self.latency_timer.run('test')
            output = self.model.decode(batch, search_width=self.config.search_width, search_mode=search_mode)
            self.latency_timer.run('test')

            if search_mode == 'prod':
                rank = (cast(torch.Tensor, output) + 1).tolist()
                batch_size = len(rank)

                for ib in range(batch_size):
                    local_rank = rank[ib]
                    group_list.extend([item_index] * len(local_rank))
                    ranks_list.extend(local_rank)
                    item_index += 1
            else:
                rank = output
                batch_size = len(rank)
                groups = batch[Map.UID_COL].tolist()
                groups = groups[:batch_size]
                group_list.extend(groups)
                ranks_list.extend(rank)

        results = self.metrics_aggregator(ranks_list, group_list)

        return results

    def evaluate(self, valid_dl, epoch):
        total_valid_steps = get_steps(valid_dl)

        self.model.model.eval()

        with torch.no_grad():
            print(f'(epoch {epoch}) validating: {self.processor.dataset_name}')

            results = self._evaluate(valid_dl, total_valid_steps, step=self.config.valid_step)
            metric_name = list(results.keys())[0]
            metric_value = results[metric_name]

            print(f'(epoch {epoch}) validation on {self.processor.dataset_name} dataset with {metric_name}: {metric_value:.4f}')

        self.model.model.train()

        action = self.monitor.push(metric_name, metric_value)

        if action is self.monitor.BEST:
            self.model.save(os.path.join(self.log_dir, f'{self.sign}.pt'))
            print(f'Saving best model to {self.log_dir}/{self.sign}.pt')

        return action
