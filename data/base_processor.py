import abc
import os.path
import random
from typing import Optional, Union, Callable

import pandas as pd

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

    BASE_STORE_DIR = 'data'

    def __init__(self, data_path):
        self.data_path = data_path
        self.store_dir = os.path.join(self.BASE_STORE_DIR, self.DATASET_NAME)
        os.makedirs(self.store_dir, exist_ok=True)

        state_path = os.path.join(self.BASE_STORE_DIR, 'state.yaml')
        self.state = ProcessorState(state_path)

        self._loaded: bool = False

        self.items: Optional[pd.DataFrame] = None
        self.users = Optional[pd.DataFrame] = None
        self.interactions: Optional[pd.DataFrame] = None

        self.item_vocab: Optional[dict] = None
        self.user_vocab: Optional[dict] = None

        self.test: Optional[pd.DataFrame] = None
        self.finetune: Optional[pd.DataFrame] = None

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
        raise NotImplemented

    def load_users(self) -> pd.DataFrame:
        raise NotImplemented

    def load_interactions(self) -> pd.DataFrame:
        raise NotImplemented

    def load(self):
        if os.path.exists(os.path.join(self.store_dir, 'items.parquet')) and \
                os.path.exists(os.path.join(self.store_dir, 'users.parquet')) and \
                os.path.exists(os.path.join(self.store_dir, 'interactions.parquet')):
            print(f'loading {self.DATASET_NAME} from cache')
            self.items = pd.read_parquet(os.path.join(self.store_dir, 'items.parquet'))
            print(f'loaded {len(self.items)} items')
            self.users = pd.read_parquet(os.path.join(self.store_dir, 'users.parquet'))
            print(f'loaded {len(self.users)} users')
            self.interactions = pd.read_parquet(os.path.join(self.store_dir, 'interactions.parquet'))
            print(f'loaded {len(self.interactions)} interactions')

            self.items = self._cast_to_string(self.items)
            self.users = self._cast_to_string(self.users)
            self.interactions = self._cast_to_string(self.interactions)
        else:
            print(f'loading {self.DATASET_NAME} from raw data')
            self.items = self.load_items()
            self.items = self._cast_to_string(self.items)
            print(f'loaded {len(self.items)} items')
            self.users = self.load_users()
            self.users = self._cast_to_string(self.users)
            print(f'loaded {len(self.users)} users')
            self.interactions = self.load_interactions()
            self.interactions = self._cast_to_string(self.interactions)
            print(f'loaded {len(self.interactions)} interactions')

            self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))
            self.users.to_parquet(os.path.join(self.store_dir, 'users.parquet'))
            self.interactions.to_parquet(os.path.join(self.store_dir, 'interactions.parquet'))

        if self.CAST_TO_STRING:
            self.users[self.HISTORY_COL] = self.users[self.HISTORY_COL].apply(
                lambda x: [str(item) for item in x]
            )

        self.item_vocab = dict(zip(self.items[self.ITEM_ID_COL], range(len(self.items))))
        self.user_vocab = dict(zip(self.users[self.USER_ID_COL], range(len(self.users))))

        if self._compress():
            print(f'Compressed {self.DATASET_NAME} data, re-run to load compressed data')
            return self.load()

        self._load_public_sets()
        return self

    def _compress(self):
        if self.state.compressed:
            return False

        user_set = set(self.interactions[self.USER_ID_COL].unique())
        old_user_size = len(self.users)

        self.users = self.users[self.users[self.USER_ID_COL].isin(user_set)]
        self.users = self.users.drop_duplicates(subset=[self.USER_ID_COL])

        print(f'Compressed users from {old_user_size} to {len(self.users)}')

        item_set = set(self.interactions[self.ITEM_ID_COL].unique())
        old_item_size = len(self.items)

        self.users[self.HISTORY_COL].apply(lambda x: [item_set.add(i) for i in x])
        self.items = self.items[self.items[self.ITEM_ID_COL].isin(item_set)].reset_index(drop=True)

        print(f'Compressed items from {old_item_size} to {len(self.items)}')

        self.users.to_parquet(os.path.join(self.store_dir, 'users.parquet'))
        self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))

        self.state.compressed = True
        self.state.write()

        return True

    def _cast_to_string(self, df: pd.DataFrame):
        if not self.CAST_TO_STRING:
            return df

        if self.ITEM_ID_COL in df.columns:
            df[self.ITEM_ID_COL] = df[self.ITEM_ID_COL].astype(str)

        if self.USER_ID_COL in df.columns:
            df[self.USER_ID_COL] = df[self.USER_ID_COL].astype(str)

        return df

    def _load_user_order(self):
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

    @staticmethod
    def _group_iterator(users, interactions):
        for u in users:
            yield interactions.get_group(u)

    def _split(self, group_iterator, count):
        df = pd.DataFrame()

        for group in group_iterator:
            for label in range(2):
                group_lbl = group[group[self.LABEL_COL] == label]
                selected_group_lbl = group_lbl.sample(n=min(self.MAX_INTERACTIONS_PER_USER // 2, len(group_lbl)),
                                                      replace=False)
                df = pd.concat([df, selected_group_lbl])

            if len(df) >= count:
                break

        return df

    def _load_public_sets(self):
        if self.test_set_valid and self.finetune_set_valid:
            print(f'Loading {self.DATASET_NAME} from cache')

            if self.NUM_TEST:
                self.test_set = pd.read_parquet(os.path.join(self.store_dir, 'test.parquet'))
                self.test_set = self._cast_to_string(self.test_set)
                print('Loaded test set')

            if self.NUM_FINETUNE:
                self.finetune_set = pd.read_parquet(os.path.join(self.store_dir, 'finetune.parquet'))
                self.finetune_set = self._cast_to_string(self.finetune_set)
                print('Loaded finetune set')

            self._loaded = True
            return

        print(f'Processing {self.DATASET_NAME} from item, user, interaction data')

        users_order = self._load_user_order()
        interactions = self.interactions.groupby(self.USER_ID_COL)

        iterator = self._group_iterator(users_order, interactions)

        if self.NUM_TEST:
            self.test_set = self._split(iterator, self.NUM_TEST)
            self.test_set.reset_index(drop=True, inplace=True)
            self.test_set.to_parquet(os.path.join(self.store_dir, 'test.parquet'))
            print(f'Generated test set with {len(self.test_set)}/{self.NUM_TEST} samples')

        if self.NUM_FINETUNE:
            self.finetune_set = self._split(iterator, self.NUM_FINETUNE)
            self.finetune_set.reset_index(drop=True, inplace=True)
            self.finetune_set.to_parquet(os.path.join(self.store_dir, 'finetune.parquet'))
            print(f'Generated finetune set with {len(self.finetune_set)}/{self.NUM_FINETUNE} samples')

        self._loaded = True

    def get_source_set(self, source):
        assert source in ['test', 'finetune', 'original'], 'source must be test, finetune or original'
        return self.interactions if source == 'original' else getattr(self, f'{source}_set')

    @staticmethod
    def _build_slicer(slicer: int):
        def _slicer(x):
            return x[:slicer] if slicer > 0 else x[slicer:]
        return _slicer

    def build_item_str(self, item_id, item_attrs: list, as_dict=False, item_self=False):
        if item_self:
            item = item_id
        else:
            item = self.items.iloc[self.item_vocab[item_id]]

        if as_dict:
            return {attr: item[attr] or '' for attr in item_attrs}

        if len(item_attrs) == 1:
            return item[item_attrs[0]]

        return ', '.join([f'{attr}: {item[attr]}' for attr in item_attrs])

    def _iterate(self, df: pd.DataFrame, slicer: Union[int, Callable], item_attrs=None, id_only=False, as_dict=False,):
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
                continue

            history_str = [self.build_item_str(item_id, item_attrs, as_dict) for item_id in history]
            candidate_str = self.build_item_str(candidate, item_attrs, as_dict)

            yield uid, candidate, history_str, candidate_str, label

    def generate(self, slicer: Union[int, Callable], item_attrs=None, source='test', id_only=False, as_dict=False, filter_func=None):
        if not self._loaded:
            raise RuntimeError('Datasets not loaded')

        source_set = self.get_source_set(source)

        if filter_func:
            source_set = filter_func(source_set)

        return self._iterate(source_set, slicer, item_attrs, id_only, as_dict)
