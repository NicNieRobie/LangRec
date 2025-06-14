import abc
import os
import random
from typing import Optional, Union, Callable

import pandas as pd

from data.base_processor import BaseProcessor
from data.compressor import Compressor


class BaseCTRProcessor(BaseProcessor, abc.ABC):
    MAX_HISTORY_PER_USER: int = 100
    MAX_INTERACTIONS_PER_USER: int = 20
    CAST_TO_STRING: bool

    BASE_STORE_DIR = 'data_store/ctr'

    def __init__(self, data_path='dataset'):
        super().__init__(data_path)

        self._loaded: bool = False
        self.items: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.interactions: Optional[pd.DataFrame] = None

        self.item_vocab: Optional[dict] = None
        self.user_vocab: Optional[dict] = None

        self.test_set: Optional[pd.DataFrame] = None
        self.finetune_set: Optional[pd.DataFrame] = None

    def load(self):
        try:
            print(f'Attempting to load {self.DATASET_NAME} from cache')
            self.items = self.loader.load_parquet('items')
            self.users = self.loader.load_parquet('users')
            self.interactions = self.loader.load_parquet('interactions')
            print(f'Loaded {len(self.items)} items, {len(self.users)} users, {len(self.interactions)} interactions')
        except Exception as e:
            print(f'Failed to load cached files: {e}. Loading raw data for {self.DATASET_NAME}...')
            self.items = self.loader.cast_df(self.load_items())
            self.users = self.loader.cast_df(self.load_users())
            self.interactions = self.loader.cast_df(self.load_interactions())
            print(f'Loaded {len(self.items)} items, {len(self.users)} users, {len(self.interactions)} interactions')

            self.loader.save_parquet('items', self.items)
            self.loader.save_parquet('users', self.users)
            self.loader.save_parquet('interactions', self.interactions)
            print(f'{self.DATASET_NAME} cached')

        if self.CAST_TO_STRING:
            self.users[self.HISTORY_COL] = self.users[self.HISTORY_COL].apply(lambda x: [str(i) for i in x])

        self.item_vocab = dict(zip(self.items[self.ITEM_ID_COL], range(len(self.items))))
        self.user_vocab = dict(zip(self.users[self.USER_ID_COL], range(len(self.users))))

        if Compressor(
            self.users, self.items, self.store_dir, self.state,
            self.USER_ID_COL, self.ITEM_ID_COL, self.HISTORY_COL,
            self.interactions
        ).compress():
            print(f'Compressed {self.DATASET_NAME} data')
            return self.load()

        self.load_public_sets()
        return self

    @staticmethod
    def _group_iterator(users, interactions):
        for u in users:
            yield interactions.get_group(u)

    def split(self, iterator, count) -> pd.DataFrame:
        df = pd.DataFrame()
        for group in iterator:
            for label in range(2):
                group_lbl = group[group[self.LABEL_COL] == label]
                n = min(self.MAX_INTERACTIONS_PER_USER // 2, len(group_lbl))
                df = pd.concat([df, group_lbl.sample(n=n, replace=False)])
            if len(df) >= count:
                break

        return df.reset_index(drop=True)

    def get_user_order(self, interactions, store_dir):
        path = os.path.join(store_dir, 'user_order.txt')
        if os.path.exists(path):
            return [line.strip() for line in open(path)]

        users = interactions[self.USER_ID_COL].unique().tolist()

        random.shuffle(users)

        with open(path, 'w') as f:
            f.write('\n'.join(users))

        return users

    def load_public_sets(self):
        if self.try_load_cached_splits():
            return

        print(f'Generating test and finetune sets from {self.DATASET_NAME}...')
        users_order = self.load_user_order()
        interactions = self.interactions.groupby(self.USER_ID_COL)

        iterator = self._group_iterator(users_order, interactions)

        if self.NUM_TEST:
            self.test_set = self.split(iterator, self.NUM_TEST)
            self.test_set.reset_index(drop=True, inplace=True)
            self.loader.save_parquet('test', self.test_set)
            print(f'Generated test set with {len(self.test_set)} samples')

        if self.NUM_FINETUNE:
            self.finetune_set = self.split(iterator, self.NUM_FINETUNE)
            self.finetune_set.reset_index(drop=True, inplace=True)
            self.loader.save_parquet('finetune', self.finetune_set)
            print(f'Generated finetune set with {len(self.finetune_set)} samples')

        self._loaded = True

    def get_source_set(self, source: str):
        assert source in ['test', 'finetune', 'original']
        return self.interactions if source == 'original' else getattr(self, f'{source}_set')

    def load_user_order(self):
        path = os.path.join(self.store_dir, 'user_order.txt')

        if os.path.exists(path):
            with open(path, 'r') as f:
                return [line.strip() for line in f]

        users = self.interactions[self.USER_ID_COL].unique().tolist()
        random.shuffle(users)

        with open(path, 'w') as f:
            for u in users:
                f.write(f'{u}\n')

        return users

    def _iterate(self, df: pd.DataFrame, slicer: Union[int, Callable], item_attrs=None, id_only=False, as_dict=False):
        if isinstance(slicer, int):
            slicer = self._build_slicer(slicer)
        item_attrs = item_attrs or self.default_attrs

        for _, row in df.iterrows():
            uid = row[self.USER_ID_COL]
            candidate = row[self.ITEM_ID_COL]
            label = row[self.LABEL_COL]

            user = self.users.iloc[self.user_vocab[uid]]
            history = slicer(user[self.HISTORY_COL])

            if id_only:
                yield uid, candidate, history, label
            else:
                history_str = [self.build_item_str(i, item_attrs, as_dict) for i in history]
                candidate_str = self.build_item_str(candidate, item_attrs, as_dict)
                yield uid, candidate, history_str, candidate_str, label

    def generate(self, slicer: Union[int, Callable], item_attrs=None, source='test', id_only=False, as_dict=False, filter_func=None):
        if not self._loaded:
            raise RuntimeError('Dataset not loaded')
        source_set = self.get_source_set(source)
        if filter_func:
            source_set = filter_func(source_set)
        return self._iterate(source_set, slicer, item_attrs, id_only, as_dict)
