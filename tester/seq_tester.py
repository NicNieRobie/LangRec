import random
from typing import cast

import torch
from tqdm import tqdm

from loader.map import Map
from loader.seq_preparer import SeqPreparer
from metrics.seq.seq_metrics_aggregator import SeqMetricsAggregator
from model.seq.base_seq_model import BaseSeqModel
from utils.dataloader import get_steps
from loguru import logger


class SeqTester:
    def __init__(self, config, processor, model):
        self.model = model
        self.processor = processor

        self.config = config

        self.num_codes = model.num_codes

    def _evaluate(self, dataloader, steps, step=1):
        search_mode = self.config.search_mode

        assert search_mode in ['prod', 'list',
                               'tree'], f'Unknown search mode {search_mode}, should be one of: prod, list, tree'

        group_list, ranks_list = [], []
        item_index = 0
        for index, batch in tqdm(enumerate(dataloader), total=steps, desc="Testing"):
            if random.random() * step > 1:
                continue

            # self.latency_timer.run('test')
            output = self.model.decode(batch, search_width=self.config.search_width, search_mode=search_mode)
            # self.latency_timer.run('test')

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

        aggregator = SeqMetricsAggregator.build_from_config(
            self.config.metrics,
            prod_mode=search_mode == 'prod'
        )
        results = aggregator(ranks_list, group_list)

        return results

    def evaluate(self):
        preparer = SeqPreparer(
            processor=self.processor,
            model=self.model,
            config=self.config
        )

        if not preparer.has_generated:
            self.processor.load()

        test_dl = preparer.load_or_generate(mode='test')

        cast(BaseSeqModel, self.model).set_code_meta(preparer.code_tree, preparer.code_map)

        total_valid_steps = get_steps(test_dl)

        self.model.model.eval()
        with torch.no_grad():
            results = self._evaluate(test_dl, total_valid_steps)
            logger.info(f'Evaluation results')
            for metric, value in results.items():
                logger.info(f'{metric}: {value:.4f}')

    def __call__(self):
        if self.config.latency:
            # TODO
            return

        self.evaluate()
