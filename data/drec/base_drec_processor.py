import abc
import os
import random
from typing import Optional, Union, Callable

import pandas as pd

from data.base_processor import BaseProcessor
from data.compressor import Compressor


class BaseDrecProcessor(BaseProcessor, abc.ABC):
    MAX_HISTORY_PER_USER: int = 100
    NEG_INTERACTIONS_PER_ITEM: int = 29
    CAST_TO_STRING: bool

    BASE_STORE_DIR = 'data_store'

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

    @property
    def test_set_valid(self):
        return os.path.exists(os.path.join(self.store_dir, 'test_drec.parquet')) or not self.test_set_required

    @property
    def finetune_set_valid(self):
        return os.path.exists(os.path.join(self.store_dir, 'finetune_drec.parquet')) or not self.finetune_set_required

    def try_load_cached_splits(self) -> bool:
        if self.test_set_valid and self.finetune_set_valid:
            print(f'Loading {self.DATASET_NAME} splits from cache')

            if self.NUM_TEST:
                self.test_set = self.loader.load_parquet('test_drec')
                print('Loaded test set')

            if self.NUM_FINETUNE:
                self.finetune_set = self.loader.load_parquet('finetune_drec')
                print('Loaded finetune set')

            self._loaded = True

            return True

        return False

    def split(self, interactions, items, store_dir, count) -> pd.DataFrame:
        """
        Select `count` items from the dataset. Here `ITEM_ID_COL` stores a set of item indexes that has
        `NEG_INTERACTIONS_PER_ITEM` negative examples and exactly 1 positive example. `LABEL_COL` stores
        the index of the positive item.
        """
        users_order = self.get_user_order(interactions, store_dir)
        interactions = interactions.groupby(self.USER_ID_COL)
        iterator = self._group_iterator(users_order, interactions)

        df = pd.DataFrame()
        for group in iterator:
            pos_ids = group[group[self.LABEL_COL] == 1]
            neg_ids = items[~items[self.ITEM_ID_COL].isin(pos_ids)]

            group_data = {}
            for _, row in pos_ids.iterrows():
                group_data[self.USER_ID_COL] = row[self.USER_ID_COL]
                group_data[self.LABEL_COL] = row[self.ITEM_ID_COL]
                group_data[self.ITEM_ID_COL] = [list(neg_ids.sample(n=self.NEG_INTERACTIONS_PER_ITEM, replace=False)[self.ITEM_ID_COL]) + [row[self.ITEM_ID_COL]]]

            df = pd.concat([df, pd.DataFrame.from_dict(group_data)])

            if len(df) >= count:
                break

        return df.reset_index(drop=True)

    def get_user_order(self, interactions, store_dir):
        path = os.path.join(store_dir, 'user_order.txt')
        if os.path.exists(path):
            return [int(line.strip()) for line in open(path)]

        users = interactions[self.USER_ID_COL].unique().tolist()

        random.shuffle(users)

        with open(path, 'w') as f:
            f.write('\n'.join(users))

        return users

    def load_public_sets(self):
        if self.try_load_cached_splits():
            return

        print(f'Generating test and finetune sets from {self.DATASET_NAME}...')

        if self.NUM_TEST:
            self.test_set = self.split(self.interactions, self.items, self.store_dir, self.NUM_TEST)
            self.loader.save_parquet('test_drec', self.test_set)
            print(f'Generated test set for DRec task with {len(self.test_set)} samples')

        if self.NUM_FINETUNE:
            self.finetune_set = self.split(self.interactions, self.items, self.store_dir, self.NUM_FINETUNE)
            self.loader.save_parquet('finetune_drec', self.finetune_set)
            print(f'Generated finetune set for DRec task with {len(self.finetune_set)} samples')

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
            candidates = row[self.ITEM_ID_COL]
            label = row[self.LABEL_COL]

            user = self.users.iloc[self.user_vocab[uid]]
            history = slicer(user[self.HISTORY_COL])

            if id_only:
                yield uid, candidates, history, label
            else:
                history_str = [self.build_item_str(i, item_attrs, as_dict) for i in history]
                candidate_str = [self.build_item_str(candidate, item_attrs, as_dict) for candidate in candidates]
                label_str = self.build_item_str(label, item_attrs, as_dict)
                yield uid, candidates, history_str, candidate_str, label_str

    def generate(self, slicer: Union[int, Callable], item_attrs=None, source='test', id_only=False, as_dict=False, filter_func=None):
        if not self._loaded:
            raise RuntimeError('Dataset not loaded')
        source_set = self.get_source_set(source)
        if filter_func:
            source_set = filter_func(source_set)
        return self._iterate(source_set, slicer, item_attrs, id_only, as_dict)
