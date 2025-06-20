import os.path

import numpy as np
import torch
from tqdm import tqdm

from loader.discrete_code_preparer import DiscreteCodePreparer
from loader.map import Map
from metrics.ctr.ctr_metrics_aggregator import CTRMetricsAggregator
from utils import bars
from utils.dataloader import get_steps
from utils.export_writer import ExportWriter
from utils.metrics import get_metrics_aggregator
from loguru import logger

from utils.timer import Timer


class CTRTester:
    def __init__(self, config, processor, model, run_name):
        self.config = config

        self.model_name = config.model.upper()
        self.dataset = config.dataset.upper()

        self.type = config.type
        assert self.type in ['prompt', 'embed'], f'Type {self.type} is not supported.'
        self.use_prompt = self.type == 'prompt'
        self.use_embed = self.type == 'embed'

        assert config.task == "ctr"
        self.task = config.task

        self.subset = "test"

        self.processor = processor
        self.model = model

        self.sign = ''

        self.export_dir = os.path.join('export', run_name)

        os.makedirs(self.export_dir, exist_ok=True)

        self.exporter = ExportWriter(os.path.join(self.export_dir))

        if self.config.rerun:
            self.exporter.reset()

        if self.use_encoding():
            self.num_codes = model.num_codes

        self.latency_timer = Timer(activate=False)

    def use_encoding(self):
        return self.config.code_path is not None or self.config.code_type is not None

    @staticmethod
    def _truncate_inputs(history, candidate, top_k_ratio=0.1):
        lengths = [len(item) for item in history]
        sorted_indices = np.argsort(lengths)[::-1].tolist()
        top_k = max(int(len(sorted_indices) * top_k_ratio), 1)

        for i in sorted_indices[:top_k]:
            history[i] = history[i][:max(len(history[i]) // 2, 10)]

        return history, candidate[:max(len(candidate) // 2, 10)]

    def _retry_with_truncation(self, history, candidate, input_template):
        self.latency_timer.run('test')
        for _ in range(5):
            for i in range(len(history)):
                _history = [f'({j + 1}) {history[i + j]}' for j in range(len(history) - i)]
                input_sequence = input_template.format('\n'.join(_history), candidate)
                response = self.model(input_sequence)

                if response is not None:
                    return response

            history, candidate = self._truncate_inputs(history, candidate)

        return None

    def test_prompt(self):
        input_template = "User behavior sequence: \n{0}\nCandidate item: {1}"

        progress = 0

        if self.exporter.scores_exist() and not self.config.latency:
            responses = self.exporter.read_scores(as_float=False)
            progress = len(responses)
            logger.debug(f'Start from {progress}')

        source_set = self.processor.get_source_set(self.subset)
        data_gen = self.processor.generate(slicer=self.config.history_window, source=self.config.source)

        for idx, data in enumerate(bar := bars.TestBar()(data_gen, total=len(source_set))):
            if idx < progress:
                continue

            uid, item_id, history, candidate, label = data

            response = self._retry_with_truncation(history, candidate, input_template)

            if response is None:
                logger.error(f'Failed to get response for {idx} ({uid}, {item_id})')
                exit(0)

            self.latency_timer.run('test')

            if isinstance(response, int):
                response = f'{response:.4f}'
            elif isinstance(response, str):
                response = response.strip(" \n")
                try:
                    # Take only the first line of the answer if the model has multiline output. This is done so that the
                    # export of the model output is not all messed up.
                    response = response.split('\n')[0]
                finally:
                    pass
            else:
                pass
            bar.set_postfix_str(f'label: {label}, response: {response}')

            self.exporter.write_scores(response)

        self.evaluate()

    def _get_embedding(self, _id, data, is_user=True):
        embed_dict = self.exporter.load_embed('user' if is_user else 'item')
        history, candidate = data['history'], data['candidate']

        template = "User behavior sequence: \n{0}" if is_user else "Candidate item: {0}"

        if _id in embed_dict:
            return embed_dict[_id], embed_dict

        embed = None
        self.latency_timer.run('test')
        if self.model.AS_DICT:
            embed = self.model.embed(history if is_user else [candidate])
        else:
            for _ in range(5):
                if is_user:
                    for i in range(len(history)):
                        _history = [f'({j + 1}) {history[i + j]}' for j in range(len(history) - i)]
                        input_seq = template.format('\n'.join(_history))
                        embed = self.model.embed(input_seq)

                        if embed is not None:
                            break
                else:
                    input_seq = template.format(candidate)
                    embed = self.model.embed(input_seq)

                    if embed is not None:
                        break

                    candidate = candidate[:len(candidate) // 2]

                if embed is not None:
                    break

                history, candidate = self._truncate_inputs(history, candidate)

        if embed is None:
            logger.error(f'Failed to get {"user" if is_user else "item"} embeddings for {_id}')
            exit(0)

        self.latency_timer.run('test')

        embed_dict[_id] = embed
        return embed, embed_dict

    def test_embed(self):
        progress = 0

        if self.exporter.scores_exist():
            responses = self.exporter.read_scores()
            progress = len(responses)
            logger.debug(f'Start testing from {progress}')

        source_set = self.processor.get_source_set(self.subset)
        data_gen = self.processor.generate(
            slicer=self.config.history_window,
            source=self.config.source,
            as_dict=self.model.AS_DICT
        )

        for idx, data in enumerate(bar := bars.TestBar()(data_gen, total=len(source_set))):
            if idx < progress:
                continue

            uid, item_id, history, candidate, label = data

            data_dict = {'history': history, 'candidate': candidate}

            user_embed, user_dict = self._get_embedding(uid, data_dict, is_user=True)
            item_embed, item_dict = self._get_embedding(item_id, data_dict, is_user=False)

            if idx % 100 == 0:
                self.exporter.save_embed('user', user_dict)
                self.exporter.save_embed('item', item_dict)

            score = torch.dot(torch.tensor(item_embed), torch.tensor(user_embed)).item()

            bar.set_postfix_str(f'label: {label}, score: {score:.4f}')

            self.exporter.write_scores(score)

        self.evaluate()

    def test_encoding(self):
        preparer = DiscreteCodePreparer(
            processor=self.processor,
            model=self.model,
            config=self.config
        )
        if not preparer.has_generated:
            self.processor.load()

        test_dl = preparer.load_or_generate(mode='test')

        total_valid_steps = get_steps(test_dl)

        self.model.model.eval()
        with torch.no_grad():
            score_list, label_list, group_list = [], [], []
            for index, batch in tqdm(enumerate(test_dl), total=total_valid_steps, desc="Testing"):
                self.latency_timer.run('test')
                scores = self.model.evaluate(batch)
                self.latency_timer.run('test')
                labels = batch[Map.LBL_COL].tolist()
                groups = batch[Map.UID_COL].tolist()

                score_list.extend(scores)
                label_list.extend(labels)
                group_list.extend(groups)

            aggregator = CTRMetricsAggregator.build_from_config(self.config.metrics)
            results = aggregator(score_list, label_list, group_list)

            logger.info(f'Evaluation results')
            for metric, value in results.items():
                logger.info(f'{metric}: {value:.4f}')

    def evaluate(self):
        scores = self.exporter.read_scores()

        source_set = self.processor.get_source_set(self.config.source)

        labels = source_set[self.processor.LABEL_COL].values
        groups = source_set[self.processor.USER_ID_COL].values

        aggregator = get_metrics_aggregator(self.task, metrics_config=self.config.metrics)

        results = aggregator(scores, labels, groups)

        logger.info(f'Evaluation results')
        for metric, value in results.items():
            logger.info(f'{metric}: {value:.4f}')

        latency_st = self.latency_timer.status_dict["test"]
        results['Avg inference time'] = latency_st.avgms()
        results['Total steps'] = latency_st.count

        self.exporter.save_metrics(results)

    def __call__(self):
        self.latency_timer.activate()
        self.latency_timer.clear()

        if self.use_encoding():
            self.test_encoding()

        if self.use_prompt:
            self.test_prompt()
        else:
            self.test_embed()

        st = self.latency_timer.status_dict["test"]
        logger.debug(f'Total {st.count} steps, avg ms {st.avgms():.4f}')
