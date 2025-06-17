import os.path
import os.path

import numpy as np
import torch
from tqdm import tqdm

from loader.discrete_code_preparer import DiscreteCodePreparer
from loader.map import Map
from loader.preparer import Preparer
from metrics.ctr.ctr_metrics_aggregator import CTRMetricsAggregator
from model.base_discrete_code_model import BaseDiscreteCodeModel
from model.base_model import BaseModel
from tuner.tune_utils.monitor import Monitor
from tuner.tuner import Tuner
from utils.code import get_code_indices
from utils.dataloader import get_steps
from utils.discovery.class_library import ClassLibrary
from utils.gpu import get_device
from utils.timer import Timer


class CTRTuner(Tuner):
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

        if self.use_encoding():
            self.PREPARER_CLASS = DiscreteCodePreparer

    def build_metrics_aggregator(self):
        return CTRMetricsAggregator.build_from_config([self.config.valid_metric])

    def load_model(self):
        models = ClassLibrary.models(self.config.task)

        if self.model_name not in models:
            raise ValueError(f'Unknown model: {self.model_name}')

        model = models[self.model_name]
        device = get_device(self.config.gpu)

        if issubclass(model, BaseDiscreteCodeModel):
            _, _, num_codes = get_code_indices(self.config, device)

            return model(device=device, num_codes=num_codes).load()
        else:
            return model(device=device).load()

    def evaluate(self, valid_dl, epoch):
        total_valid_steps = get_steps(valid_dl)

        self.model.model.eval()
        with torch.no_grad():
            metric_name, metric_values = None, []
            print(f'[Epoch {epoch}] Validating on dataset {self.processor.dataset_name}')
            score_list, label_list, group_list = [], [], []
            for index, batch in enumerate(tqdm(valid_dl, total=total_valid_steps, desc="Validating")):
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
