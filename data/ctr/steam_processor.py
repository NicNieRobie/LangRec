import os.path

import pandas as pd

from data.ctr.neg_sampling_processor import NegSamplingProcessor
from data.ctr.seq_to_label_processor import SequenceToLabelProcessor


class SteamProcessor(NegSamplingProcessor, SequenceToLabelProcessor):
    DATASET_NAME = 'steam'

    ITEM_ID_COL = 'app_id'
    USER_ID_COL = 'user_id'
    HISTORY_COL = 'history'
    LABEL_COL = 'click'

    POS_SAMPLE_COUNT = 2
    NEG_RATIO = 2

    NUM_TEST = 10_000
    NUM_FINETUNE = 80_000

    CAST_TO_STRING = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._interactions = None

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_path, 'steam.csv')

        steam = pd.read_csv(path, sep=',', encoding='ISO-8859-1')

        steam['title'] = steam['title'].str.replace(r'[^A-Za-z0-9 ]+', '')

        return steam

    def load_users(self) -> pd.DataFrame:
        path = os.path.join(self.data_path, 'ratings.csv')

        interactions = pd.read_csv(path, sep=',')

        interactions = interactions[interactions[self.RATING_COL] != 3].copy()
        interactions[self.LABEL_COL] = interactions[self.RATING_COL] > 3
        interactions.drop(columns=[self.RATING_COL], inplace=True)

        return self._load_users(interactions)
