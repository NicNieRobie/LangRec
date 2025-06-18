import abc
import random

import pandas as pd

from data.ctr.base_ctr_processor import BaseCTRProcessor
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class NegSamplingProcessor(BaseCTRProcessor, abc.ABC):
    NEG_RATIO: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._user_dict = None
        self._pos_inters = None

    def _get_user_dict_from_interactions(self, interactions):
        self._user_dict = dict()
        for idx, row in interactions.iterrows():
            user_id = row[self.USER_ID_COL]
            item_id = row[self.ITEM_ID_COL]

            if user_id not in self._user_dict:
                self._user_dict[user_id] = set()

            self._user_dict[user_id].add(item_id)

    def load_interactions(self) -> pd.DataFrame:
        assert self._user_dict is not None and self._pos_inters is not None, 'User dictionary and positive interactions have to be generated first'

        user_set = set(self.users[self.USER_ID_COL].unique())

        items = self.items[self.ITEM_ID_COL].unique().tolist()
        num_items = len(items)
        interactions = []

        for user_id in self._user_dict:
            if user_id not in user_set:
                continue

            user_interactions = self._user_dict[user_id]

            for _ in range(self.NEG_RATIO):
                neg_item = items[random.randint(0, num_items - 1)]

                while neg_item in user_interactions:
                    neg_item = items[random.randint(0, num_items - 1)]

                interactions.append({self.USER_ID_COL: user_id, self.ITEM_ID_COL: neg_item, self.LABEL_COL: 0})

        interactions = pd.DataFrame(interactions)
        interactions = pd.concat([interactions, self._pos_inters], ignore_index=True)

        return interactions
