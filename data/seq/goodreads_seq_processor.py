import os

import pandas as pd

from data.seq.timed_interaction_seq_processor import TimedInteractionSeqProcessor


class GoodReadsProcessor(TimedInteractionSeqProcessor):
    DATASET_NAME = 'goodreads'

    ITEM_ID_COL = 'bid'
    USER_ID_COL = 'uid'
    HISTORY_COL = 'history'
    LABEL_COL = 'click'
    DATE_COL = 'timestamp'

    RATING_COL = 'rating'

    POS_SAMPLE_COUNT = 2

    NUM_TEST = 4_000
    NUM_FINETUNE = 15_000

    CAST_TO_STRING = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._interactions = None

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_path, 'good.csv')

        books = pd.read_csv(path, sep=',', encoding='ISO-8859-1')

        books['title'] = books['title'].str.replace(r'[^A-Za-z0-9 ]+', '')

        return books

    def load_users(self) -> pd.DataFrame:
        path = os.path.join(self.data_path, 'ratings.csv')

        interactions = pd.read_csv(path, sep=',')

        interactions = interactions[interactions[self.RATING_COL] != 3].copy()
        interactions[self.LABEL_COL] = interactions[self.RATING_COL] > 3
        interactions.drop(columns=[self.RATING_COL], inplace=True)

        return self._load_users(interactions)

    def load_interactions(self) -> pd.DataFrame:
        pass
