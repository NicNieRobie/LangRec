import random
from typing import cast

import torch
from tqdm import tqdm

from loader.map import Map
from loader.drec_preparer import DrecPreparer
from metrics.drec.drec_metrics_aggregator import DrecMetricsAggregator
from model.drec.base_drec_model import BaseDrecModel
from utils.dataloader import get_steps
from utils.timer import Timer

from loguru import logger


class DrecTester:
    def __init__(self, config, processor, model):
        self.model = model
        self.processor = processor

        self.config = config

        self.num_codes = model.num_codes

        self.latency_timer = Timer(activate=False)

    def _evaluate(self, dataloader, steps, step=1):
        search_mode = self.config.search_mode

        assert search_mode in ['prod', 'list',
                               'tree'], f'Unknown search mode {search_mode}, should be one of: prod, list, tree'

        group_list, ranks_list = [], []
        item_index = 0
        for index, batch in tqdm(enumerate(dataloader), total=steps, desc="Testing"):
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

        aggregator = DrecMetricsAggregator.build_from_config(
            self.config.metrics,
            prod_mode=search_mode == 'prod'
        )
        results = aggregator(ranks_list, group_list)

        latency_st = self.latency_timer.status_dict["test"]
        results['Avg inference time'] = latency_st.avgms()
        results['Total steps'] = latency_st.count

        return results

    def evaluate(self):
        preparer = DrecPreparer(
            processor=self.processor,
            model=self.model,
            config=self.config
        )

        if not preparer.has_generated:
            self.processor.load()

        test_dl = preparer.load_or_generate(mode='test')

        cast(BaseDrecModel, self.model).set_code_meta(preparer.code_tree, preparer.code_map)

        total_valid_steps = get_steps(test_dl)

        self.model.model.eval()
        with torch.no_grad():
            results = self._evaluate(test_dl, total_valid_steps)
            for metric, value in results.items():
                print(f'{metric}: {value:.4f}')

    def __call__(self):
        self.latency_timer.activate()
        self.latency_timer.clear()

        self.evaluate()

        st = self.latency_timer.status_dict["test"]
        logger.debug(f'Total {st.count} steps, avg ms {st.avgms():.4f}')
