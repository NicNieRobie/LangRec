import abc
import os
import random
from typing import Optional, Union, Callable

import pandas as pd
from tqdm import tqdm

from data.base_processor import BaseProcessor
from data.compressor import Compressor
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class BaseSeqProcessor(BaseProcessor, abc.ABC):
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
            print(f'Loaded {len(self.items)} items, {len(self.users)} users')
        except Exception as e:
            print(f'Failed to load cached files: {e}. Loading raw data for {self.DATASET_NAME}...')
            self.items = self.loader.cast_df(self.load_items())
            self.users = self.loader.cast_df(self.load_users())
            print(f'Loaded {len(self.items)} items, {len(self.users)} users')

            self.loader.save_parquet('items', self.items)
            self.loader.save_parquet('users', self.users)
            print(f'{self.DATASET_NAME} cached')

        if self.CAST_TO_STRING:
            self.users[self.HISTORY_COL] = self.users[self.HISTORY_COL].apply(lambda x: [str(i) for i in x])

        self.item_vocab = dict(zip(self.items[self.ITEM_ID_COL], range(len(self.items))))
        self.user_vocab = dict(zip(self.users[self.USER_ID_COL], range(len(self.users))))

        if Compressor(
            self.users, self.items, self.store_dir, self.state,
            self.USER_ID_COL, self.ITEM_ID_COL, self.HISTORY_COL
        ).compress():
            print(f'Compressed {self.DATASET_NAME} data')
            return self.load()

        self.load_public_sets()
        return self

    def _iterator(self, user_order, users):
        for uid in user_order:
            user = users[users[self.USER_ID_COL] == uid]
            yield user.iloc[0]

    @staticmethod
    def split(iterator, count):
        users = []

        for user in tqdm(iterator, total=count):
            users.append(user)
            if len(users) >= count:
                break

        users = pd.DataFrame(users)
        return users

    @property
    def test_set_valid(self):
        return os.path.exists(os.path.join(self.store_dir, 'test_seq.parquet')) or not self.test_set_required

    @property
    def finetune_set_valid(self):
        return os.path.exists(os.path.join(self.store_dir, 'finetune_seq.parquet')) or not self.finetune_set_required

    def load_user_order(self):
        path = os.path.join(self.store_dir, 'user_order_seq.txt')
        if os.path.exists(path):
            return [line.strip() for line in open(path)]

        users = self.users[self.USER_ID_COL].unique().tolist()
        random.shuffle(users)

        with open(path, 'w') as f:
            users_str = [str(u) for u in users]
            f.write('\n'.join(users_str))

        return users

    def load_public_sets(self):
        if self.try_load_cached_splits(suffix="_seq"):
            return

        print(f'Processing data from {self.DATASET_NAME}...')

        users_order = self.load_user_order()
        print('users cols', self.users.columns)
        iterator = self._iterator(users_order, self.users)

        if self.NUM_TEST:
            self.test_set = self.split(iterator, self.NUM_TEST)
            self.test_set.reset_index(drop=True, inplace=True)
            self.loader.save_parquet('test_seq', self.test_set)
            print(f'Generated test set with {len(self.test_set)}/{self.NUM_TEST} samples')

        if self.NUM_FINETUNE:
            self.finetune_set = self.split(iterator, self.NUM_FINETUNE)
            self.finetune_set.reset_index(drop=True, inplace=True)
            self.loader.save_parquet('finetune_seq', self.finetune_set)
            print(f'Generated finetune set with {len(self.finetune_set)}/{self.NUM_FINETUNE} samples')

        self._loaded = True

    def get_source_set(self, source: str):
        assert source in ['test', 'finetune', 'original']
        return self.users if source == 'original' else getattr(self, f'{source}_set')

    def _iterate(self, df: pd.DataFrame, slicer: Union[int, Callable], **kwargs):
        if isinstance(slicer, int):
            slicer = self._build_slicer(slicer)

        print('cols', df.columns)

        for _, row in df.iterrows():
            uid = row[self.USER_ID_COL]
            history = slicer(row[self.HISTORY_COL])

            yield uid, history

    def generate(self, slicer: Union[int, Callable], source='test', filter_func=None, **kwargs):
        if not self._loaded:
            raise RuntimeError('Dataset not loaded')

        source_set = self.get_source_set(source)

        if filter_func:
            source_set = filter_func(source_set)

        return self._iterate(source_set, slicer)
