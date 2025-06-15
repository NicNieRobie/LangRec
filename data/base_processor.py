import abc
import os
import random
from typing import Optional, Union, Callable

import pandas as pd

from data.loader import Loader
from data.processor_state import ProcessorState


class BaseProcessor(abc.ABC):
    DATASET_NAME: str
    ITEM_ID_COL: str
    USER_ID_COL: str
    HISTORY_COL: str
    LABEL_COL: str

    NUM_TEST: int
    NUM_FINETUNE: int

    MAX_HISTORY_PER_USER: int = 100
    MAX_INTERACTIONS_PER_USER: int = 20
    CAST_TO_STRING: bool

    BASE_STORE_DIR = 'data_store'

    def __init__(self, data_path='dataset'):
        self.data_path = data_path or 'dataset'
        self.store_dir = os.path.join(self.BASE_STORE_DIR, self.DATASET_NAME)
        os.makedirs(self.store_dir, exist_ok=True)

        self.state = ProcessorState(os.path.join(self.store_dir, 'state.yaml'))

        self.loader = Loader(
            base_dir=self.BASE_STORE_DIR,
            dataset_name=self.DATASET_NAME,
            cast_to_string=self.CAST_TO_STRING,
            item_id_col=self.ITEM_ID_COL,
            user_id_col=self.USER_ID_COL
        )

        self._loaded: bool = False
        self.items: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.interactions: Optional[pd.DataFrame] = None

        self.item_vocab: Optional[dict] = None
        self.user_vocab: Optional[dict] = None

        self.test_set: Optional[pd.DataFrame] = None
        self.finetune_set: Optional[pd.DataFrame] = None

    @property
    def dataset_name(self):
        return self.DATASET_NAME

    @property
    def test_set_required(self):
        return self.NUM_TEST > 0

    @property
    def finetune_set_required(self):
        return self.NUM_FINETUNE > 0

    @property
    def test_set_valid(self):
        return os.path.exists(os.path.join(self.store_dir, 'test.parquet')) or not self.test_set_required

    @property
    def finetune_set_valid(self):
        return os.path.exists(os.path.join(self.store_dir, 'finetune.parquet')) or not self.finetune_set_required

    @property
    def default_attrs(self):
        raise None

    def load_items(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_users(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_interactions(self) -> pd.DataFrame:
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def load_public_sets(self):
        raise NotImplementedError

    def get_source_set(self, source: str):
        raise NotImplementedError

    def load_user_order(self):
        raise NotImplementedError

    def _iterate(self, df: pd.DataFrame, slicer: Union[int, Callable], item_attrs=None, id_only=False, as_dict=False):
        raise NotImplementedError

    def generate(self, slicer: Union[int, Callable], item_attrs=None, source='test', id_only=False, as_dict=False,
                 filter_func=None):
        raise NotImplementedError

    @staticmethod
    def _build_slicer(slicer: int):
        def _slicer(x):
            return x[:slicer] if slicer > 0 else x[slicer:]

        return _slicer

    def build_item_str(self, item_id, item_attrs: list, as_dict=False, item_self=False):
        item = item_id if item_self else self.items.iloc[self.item_vocab[item_id]]
        if as_dict:
            return {attr: item.get(attr, '') for attr in item_attrs}
        if len(item_attrs) == 1:
            return item[item_attrs[0]]
        return ', '.join([f'{attr}: {item[attr]}' for attr in item_attrs])

    def iterate(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='original')

    def test(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='test')

    def finetune(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='finetune')

    def try_load_cached_splits(self, suffix: str = None) -> bool:
        if self.test_set_valid and self.finetune_set_valid:
            print(f'Loading {self.DATASET_NAME} splits from cache')

            if self.NUM_TEST:
                self.test_set = self.loader.load_parquet('test' + suffix)
                print('Loaded test set')

            if self.NUM_FINETUNE:
                self.finetune_set = self.loader.load_parquet('finetune' + suffix)
                print('Loaded finetune set')

            self._loaded = True

            return True

        return False

    def organize_item(self, item_id, item_attrs: list, as_dict=False, item_self=False):
        if item_self:
            item = item_id
        else:
            item = self.items.iloc[self.item_vocab[item_id]]

        if as_dict:
            return {attr: item[attr] or '' for attr in item_attrs}

        if len(item_attrs) == 1:
            return item[item_attrs[0]]

        return ', '.join([f'{attr}: {item[attr]}' for attr in item_attrs])

    def get_item_subset(self, source, slicer: Union[int, Callable]):
        item_set = set()

        if isinstance(slicer, int):
            slicer = self._build_slicer(slicer)

        source_set = self.get_source_set(source)
        for _, row in source_set.iterrows():
            user_id = row[self.USER_ID_COL]
            item_id = row[self.ITEM_ID_COL]

            user = self.users.iloc[self.user_vocab[user_id]]
            history = slicer(user[self.HISTORY_COL])

            item_set.add(item_id)
            item_set.update(history)

        return item_set

    def load_valid_user_set(self, valid_ratio: float, task: str) -> set:
        path = os.path.join(self.store_dir, f'valid_user_set_{valid_ratio}_{task}.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return {line.strip() for line in f}

        users = self.finetune_set[self.USER_ID_COL].unique().tolist()
        random.shuffle(users)

        valid_user_num = int(valid_ratio * len(users))
        valid_user_set = users[:valid_user_num]

        with open(path, 'w') as f:
            for u in valid_user_set:
                f.write(f'{u}\n')

        return set(map(str, valid_user_set))
