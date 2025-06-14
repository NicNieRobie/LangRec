import os

import numpy as np
from tqdm import tqdm

from utils.discovery.class_library import ClassLibrary
from utils.load_processor import load_processor


class Embedder:
    def __init__(self, dataset, model, task, attrs, device):
        self.device = device

        self.data = dataset.upper()
        self.model_name = model.upper()

        self.attrs = attrs

        self.task = task.lower()

        self.processor = load_processor(self.data, self.task)
        self.processor.load()

        self.model = self._load_model()

        self.log_dir = os.path.join('export', self.data)

        os.makedirs(self.log_dir, exist_ok=True)

        self.embedding_path = os.path.join(self.log_dir, f'{self.model_name}-embeds-{self.task}.npy')

    def _load_model(self):
        models = ClassLibrary.models()

        if self.model_name not in models:
            raise ValueError(f'Unknown model: {self.model_name}')

        model = models[self.model_name]

        return model(device=self.device).load()

    def _embed(self):
        if os.path.exists(self.embedding_path):
            print('Embeddings file exists, skipping embedding...')
            return self.embedding_path

        item_embeddings = []

        for item_id in tqdm(self.processor.item_vocab):
            item = self.processor.organize_item(item_id, item_attrs=self.attrs or self.processor.default_attrs)
            embedding = self.model.embed(item or '[Empty]', truncate=True)
            item_embeddings.append(embedding)

        item_embeddings = np.array(item_embeddings)

        np.save(self.embedding_path, item_embeddings)

        print(f'Embeddings saved to {self.embedding_path}')

        return self.embedding_path

    def run(self):
        return self._embed()
