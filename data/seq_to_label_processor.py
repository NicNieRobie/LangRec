import abc

import pandas as pd

from data.base_processor import BaseProcessor
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class SequenceToLabelProcessor(BaseProcessor, abc.ABC):
    POS_SAMPLE_COUNT: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pos_interactions = None

    def _extract_pos_interactions(self, users: pd.DataFrame):
        users = users[users[self.HISTORY_COL].apply(len) > self.POS_SAMPLE_COUNT]

        self._pos_interactions = pd.DataFrame([
            {
                self.USER_ID_COL: row[self.USER_ID_COL],
                self.ITEM_ID_COL: row[self.HISTORY_COL][-(i + 1)],
                self.LABEL_COL: 1
            }
            for _, row in users.iterrows()
            for i in range(self.POS_SAMPLE_COUNT)
        ])

        users = users.assign(
            **{self.HISTORY_COL: users[self.HISTORY_COL].apply(
                lambda x: x[-self.MAX_HISTORY_PER_USER - self.POS_SAMPLE_COUNT: -self.POS_SAMPLE_COUNT]
            )}
        )

        return users
