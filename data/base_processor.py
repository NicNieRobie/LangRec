import abc
import os
from typing import Optional, Union, Callable

import pandas as pd

from data.processor_state import ProcessorState
from data.loader import Loader
from data.compressor import Compressor
from data.splitter import Splitter


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

        self.splitter = Splitter(
            user_id_col=self.USER_ID_COL,
            label_col=self.LABEL_COL,
            max_interactions=self.MAX_INTERACTIONS_PER_USER
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
        raise NotImplementedError

    def load_items(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_users(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_interactions(self) -> pd.DataFrame:
        raise NotImplementedError

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
            self.users, self.items, self.interactions,
            self.store_dir, self.state,
            self.USER_ID_COL, self.ITEM_ID_COL, self.HISTORY_COL
        ).compress():
            print(f'Compressed {self.DATASET_NAME} data')
            return self.load()

        self._load_public_sets()
        return self

    def _load_public_sets(self):
        if self.test_set_valid and self.finetune_set_valid:
            print(f'Loading {self.DATASET_NAME} splits from cache')
            if self.NUM_TEST:
                self.test_set = self.loader.load_parquet('test')
                print('Loaded test set')
            if self.NUM_FINETUNE:
                self.finetune_set = self.loader.load_parquet('finetune')
                print('Loaded finetune set')
            self._loaded = True
            return

        print(f'Generating test and finetune sets from {self.DATASET_NAME}...')
        users_order = self.splitter.get_user_order(self.interactions, self.store_dir)

        if self.NUM_TEST:
            self.test_set = self.splitter.split(self.interactions, users_order, self.NUM_TEST)
            self.loader.save_parquet('test', self.test_set)
            print(f'Generated test set with {len(self.test_set)} samples')

        if self.NUM_FINETUNE:
            self.finetune_set = self.splitter.split(self.interactions, users_order, self.NUM_FINETUNE)
            self.loader.save_parquet('finetune', self.finetune_set)
            print(f'Generated finetune set with {len(self.finetune_set)} samples')

        self._loaded = True

    def get_source_set(self, source: str):
        assert source in ['test', 'finetune', 'original']
        return self.interactions if source == 'original' else getattr(self, f'{source}_set')

    def _build_slicer(self, slicer: int):
        def _slicer(x):
            return x[:slicer] if slicer > 0 else x[slicer:]
        return _slicer

    def build_item_str(self, item_id, item_attrs: list, as_dict=False, item_self=False):
        item = item_id if item_self else self.items.iloc[self.item_vocab[item_id]]
        if as_dict:
            return {attr: item.get(attr, '') for attr in item_attrs}
        if len(item_attrs) == 1:
            return item[item_attrs[0]]
        return ', '.join(f'{attr}: {item[attr]}' for attr in item_attrs)

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
