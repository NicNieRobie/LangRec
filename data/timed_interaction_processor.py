import abc

import pandas as pd

from data.seq_to_label_processor import SequenceToLabelProcessor
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class TimedInteractionProcessor(SequenceToLabelProcessor, abc.ABC):
    DATE_COL: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._interactions = None

    def _load_users(self, interactions):
        item_id_set = set(self.items[self.ITEM_ID_COL].unique())

        interactions = interactions[interactions[self.ITEM_ID_COL].isin(item_id_set)]
        interactions = (
            interactions
            .groupby(self.USER_ID_COL)
            .filter(lambda df: df[self.LABEL_COL].nunique() == 2)
        )

        self._interactions = interactions

        pos_interactions = interactions[interactions[self.LABEL_COL] == 1]

        users = (
            pos_interactions.sort_values([self.USER_ID_COL, self.DATE_COL])
            .groupby(self.USER_ID_COL)[self.ITEM_ID_COL]
            .apply(list)
            .reset_index(name=self.HISTORY_COL)
        )

        return self._extract_pos_interactions(users)

    def load_interactions(self) -> pd.DataFrame:
        uid_set = set(self.users[self.USER_ID_COL].nunique())

        neg_interactions = (
            self._interactions[self._interactions[self.LABEL_COL] == 0]
            .loc[lambda df: df[self.USER_ID_COL].isin(uid_set)]
            .drop(columns=[self.DATE_COL])
        )

        return pd.concat([neg_interactions, self._pos_interactions], ignore_index=True)
