import abc
import os
import random
from typing import Optional, Union, Callable

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

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
            logger.debug(f'Attempting to load {self.DATASET_NAME} from cache')
            self.items = self.loader.load_parquet('items')
            self.users = self.loader.load_parquet('users')
            self.interactions = self.loader.load_parquet('interactions')
            logger.debug(f'Loaded {len(self.items)} items, {len(self.users)} users, {len(self.interactions)} interactions')
        except Exception as e:
            logger.debug(f'Failed to load cached files: {e}. Loading raw data for {self.DATASET_NAME}...')
            self.items = self.loader.cast_df(self.load_items())
            self.users = self.loader.cast_df(self.load_users())
            self.interactions = self.loader.cast_df(self.load_interactions())
            logger.debug(f'Loaded {len(self.items)} items, {len(self.users)} users, {len(self.interactions)} interactions')

            self.loader.save_parquet('items', self.items)
            self.loader.save_parquet('users', self.users)
            self.loader.save_parquet('interactions', self.interactions)
            logger.debug(f'{self.DATASET_NAME} cached')

        self.interactions = self.interactions[
            self.interactions[self.USER_ID_COL].isin(self.users[self.USER_ID_COL])
        ]

        self.interactions = self.interactions[
            self.interactions[self.ITEM_ID_COL].isin(self.items[self.ITEM_ID_COL])
        ]

        if self.CAST_TO_STRING:
            self.users[self.HISTORY_COL] = self.users[self.HISTORY_COL].apply(lambda x: [str(i) for i in x])

        self.item_vocab = dict(zip(self.items[self.ITEM_ID_COL], range(len(self.items))))
        self.user_vocab = dict(zip(self.users[self.USER_ID_COL], range(len(self.users))))

        if Compressor(
            self.users, self.items, self.store_dir, self.state,
            self.USER_ID_COL, self.ITEM_ID_COL, self.HISTORY_COL,
            self.interactions
        ).compress():
            logger.debug(f'Compressed {self.DATASET_NAME} data')
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

    def split(self, iterator, items, count) -> pd.DataFrame:
        """
        Select `count` items from the dataset. Here `ITEM_ID_COL` stores a set of item indexes that has
        `NEG_INTERACTIONS_PER_ITEM` negative examples and exactly 1 positive example. `LABEL_COL` stores
        the index of the positive item.
        """
        df = pd.DataFrame()
        with tqdm(total=count, desc="Generating split") as pbar:
            for group in iterator:
                pos_ids = group[group[self.LABEL_COL] == 1]

                if len(pos_ids) == 0:
                    continue

                neg_ids = items[~items[self.ITEM_ID_COL].isin(pos_ids)]

                true_label = pos_ids[self.ITEM_ID_COL].iloc[-1]

                group_data = {
                    self.USER_ID_COL: pos_ids[self.USER_ID_COL].iloc[-1],
                    self.LABEL_COL: true_label,
                    self.ITEM_ID_COL: [
                        list(
                            neg_ids.sample(n=self.NEG_INTERACTIONS_PER_ITEM, replace=False)[self.ITEM_ID_COL]
                        ) + [true_label]
                    ],
                }

                df = pd.concat([df, pd.DataFrame.from_dict(group_data)])

                if len(df) % 100 == 0:
                    pbar.update(100)

                if len(df) >= count:
                    break

        return df.reset_index(drop=True)

    def load_public_sets(self):
        if self.try_load_cached_splits(suffix="_drec"):
            return

        logger.debug(f'Generating test and finetune sets from {self.DATASET_NAME}...')
        users_order = self.load_user_order()
        interactions = self.interactions.groupby(self.USER_ID_COL)

        iterator = self._group_iterator(users_order, interactions)

        if self.NUM_TEST:
            self.test_set = self.split(iterator, self.items, self.NUM_TEST)
            self.test_set.reset_index(drop=True, inplace=True)
            self.loader.save_parquet('test_drec', self.test_set)
            logger.debug(f'Generated test set for DRec task with {len(self.test_set)} samples')

        if self.NUM_FINETUNE:
            self.finetune_set = self.split(iterator, self.items, self.NUM_FINETUNE)
            self.finetune_set.reset_index(drop=True, inplace=True)
            self.loader.save_parquet('finetune_drec', self.finetune_set)
            logger.debug(f'Generated finetune set for DRec task with {len(self.finetune_set)} samples')

        self._loaded = True

    def get_source_set(self, source: str):
        assert source in ['test', 'finetune', 'original']
        return self.interactions if source == 'original' else getattr(self, f'{source}_set')

    def load_user_order(self):
        path = os.path.join(self.store_dir, 'user_order_drec.txt')
        if os.path.exists(path):
            return [line.strip() for line in open(path)]

        users = self.interactions[self.USER_ID_COL].unique().tolist()
        random.shuffle(users)

        with open(path, 'w') as f:
            users_str = [str(u) for u in users]
            f.write('\n'.join(users_str))

        return users

    def _iterate(self, df: pd.DataFrame, slicer: Union[int, Callable], item_attrs=None, id_only=False, as_dict=False):
        if isinstance(slicer, int):
            slicer = self._build_slicer(slicer)
        item_attrs = item_attrs or self.default_attrs

        for _, row in df.iterrows():
            uid = row[self.USER_ID_COL]
            candidates = row[self.ITEM_ID_COL].astype("int64")
            label = np.int64(row[self.LABEL_COL])

            user = self.users.iloc[self.user_vocab[uid]]
            history = slicer(user[self.HISTORY_COL].astype("int64"))
            # history and candidate items are converted to lists of ints. This done so that torch will not shit itself
            # as soon as `loader.Datasets`'s __getitem__ is used.

            if id_only:
                yield uid, candidates, history, label
            else:
                history_str = [self.build_item_str(i, item_attrs, as_dict) for i in history]
                candidates_str = [self.build_item_str(candidate, item_attrs, as_dict) for candidate in candidates]
                label_str = self.build_item_str(label, item_attrs, as_dict)
                yield uid, candidates, history_str, candidates_str, label_str

    def generate(self, slicer: Union[int, Callable], item_attrs=None, source='test', id_only=False, as_dict=False, filter_func=None):
        if not self._loaded:
            raise RuntimeError('Dataset not loaded')
        source_set = self.get_source_set(source)
        if filter_func:
            source_set = filter_func(source_set)
        return self._iterate(source_set, slicer, item_attrs, id_only, as_dict)
