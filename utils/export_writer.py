import json
import os.path
from typing import Dict

import numpy as np


class ExportWriter:
    def __init__(self, export_dir):
        self.export_dir = export_dir

        self.scores_path = os.path.join(self.export_dir, 'scores_data.txt')
        self.metrics_path = os.path.join(self.export_dir, 'metrics.json')
        self.embed_path = os.path.join(self.export_dir, 'embeds_{0}.npy')

    def reset(self):
        if os.path.exists(self.export_dir):
            os.remove(self.export_dir)

    def scores_exist(self):
        return os.path.exists(self.scores_path)

    def write_scores(self, data):
        with open(self.scores_path, 'a') as f:
            f.write(f'{data}\n')

    def read_scores(self, as_float=True):
        path = self.scores_path
        transform = float if as_float else str

        with open(path, 'r') as f:
            return [transform(line.strip()) for line in f]

    def save_metrics(self, metrics_dict: dict):
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    def load_embed(self, entity):
        if not os.path.exists(self.embed_path.format(entity)):
            return {}

        try:
            entity_arr = np.load(self.embed_path.format(entity), allow_pickle=True)
            return entity_arr.item()
        except EOFError:
            return {}

    def save_embed(self, entity, embed_dict: Dict[str, np.ndarray]):
        np.save(self.embed_path.format(entity), embed_dict, allow_pickle=True)
