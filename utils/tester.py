import os.path

import numpy as np
import torch

from utils import bars
from utils.exporter import Exporter

from metrics.metrics_aggregator import MetricsAggregator


class Tester:
    def __init__(self, config, processor, model):
        self.config = config

        self.model_name = config.model.upper()
        self.dataset = config.dataset.upper()

        self.type = config.type
        assert self.type in ['prompt', 'embed'], f'Type {self.type} is not supported.'
        self.use_prompt = self.type == 'prompt'
        self.use_embed = self.type == 'embed'

        assert config.task in ["ctr", "drec", "seq"]
        self.task = config.task

        self.subset = "test"

        self.processor = processor
        self.model = model

        self.sign = ''

        self.log_dir = os.path.join('export', self.dataset)

        if self.use_embed:
            assert self.config.embed_func in ['last', 'pool']
            embed_suffix = '_embed'
            if self.config.embed_func == 'pool':
                embed_suffix += '_pool'
            self.log_dir = os.path.join('export', self.dataset + embed_suffix)

        os.makedirs(self.log_dir, exist_ok=True)

        self.exporter = Exporter(os.path.join(self.log_dir, f'{self.model_name}{self.sign}_{self.task}.dat'))

        if self.config.rerun:
            self.exporter.reset()

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

    def test_prompt(self):
        if self.config.task in ["drec", "seq"]:
            input_template = "User behavior sequence: \n{0}\nCandidate items: {1}"
        else:
            input_template = "User behavior sequence: \n{0}\nCandidate item: {1}"

        progress = 0

        if self.exporter.exists() and not self.config.latency:
            responses = self.exporter.read(as_float=False)
            progress = len(responses)
            print(f'Start from {progress}')

        source_set = self.processor.get_source_set(self.subset)
        data_gen = self.processor.generate(slicer=self.config.history_window, source=self.config.source)

        for idx, data in enumerate(bar := bars.TestBar()(data_gen, total=len(source_set))):
            if idx < progress:
                continue

            uid, item_id, history, candidate, label = data # candidate is a list of candidates in case of drec

            if self.task == "drec":
                candidate = "\n".join(candidate)

            response = self._retry_with_truncation(history, candidate, input_template)

            if response is None:
                print(f'Failed to get response for {idx} ({uid}, {item_id})')
                exit(0)

            if isinstance(response, int):
                response = f'{response:.4f}'
            elif isinstance(response, str):
                response = response.strip(" \n")
            else:
                pass
            bar.set_postfix_str(f'label: {label}, response: {response}')

            if not self.config.latency:
                self.exporter.write(response)

    def _get_embedding(self, _id, data, is_user=True):
        embed_dict = self.exporter.load_embed('user' if is_user else 'item')
        history, candidate = data['history'], data['candidate']

        template = "User behavior sequence: \n{0}" if is_user else "Candidate item: {0}"

        if _id in embed_dict:
            return embed_dict[_id], embed_dict

        embed = None
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
            print(f'Failed to get {"user" if is_user else "item"} embeddings for {_id}')
            exit(0)

        embed_dict[_id] = embed
        return embed, embed_dict

    def test_embed(self):
        progress = 0

        if self.exporter.exists():
            responses = self.exporter.read()
            progress = len(responses)
            print(f'Start from {progress}')

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

            self.exporter.write(score)

    def evaluate(self):
        scores = self.exporter.read()

        source_set = self.processor.get_source_set(self.config.source)

        labels = source_set[self.processor.LABEL_COL].values
        groups = source_set[self.processor.USER_ID_COL].values

        aggregator = MetricsAggregator.build_from_config(self.config.metrics)

        results = aggregator(scores, labels, groups)

        for metric, val in results.items():
            print(f'{metric}: {val:.4f}')

        self.exporter.save_metrics(results)

    def __call__(self):
        if self.config.latency:
            # TODO
            return

        if self.use_prompt:
            self.test_prompt()
        else:
            self.test_embed()

        self.evaluate()
