import os.path
from typing import Dict

import numpy as np


class Exporter:
    def __init__(self, export_path):
        self.export_path = export_path
        self.metrics_path = export_path + '.metrics'
        self.embed_path = export_path + '.{0}.npy'
        self.convert_path = export_path + '.convert'

    def reset(self):
        if os.path.exists(self.export_path):
            os.remove(self.export_path)

        if os.path.exists(self.convert_path):
            os.remove(self.convert_path)

    def exists(self):
        return os.path.exists(self.export_path)

    def write(self, data):
        with open(self.export_path, 'a') as f:
            f.write(f'{data}\n')

    def read(self, as_float=True):
        path = self.export_path
        transform = float if as_float else str

        with open(path, 'r') as f:
            return [transform(line.strip()) for line in f]

    def save_metrics(self, metrics_dict: dict):
        with open(self.metrics_path, 'w') as f:
            for metric, val in metrics_dict.items():
                f.write(f'{metric}: {val:.4f}\n')

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

    def save_convert(self, scores):
        with open(self.convert_path, 'w') as f:
            for score in scores:
                f.write(f'{score}\n')
