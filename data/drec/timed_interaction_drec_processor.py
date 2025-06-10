import abc

from data.drec.base_drec_processor import BaseDrecProcessor
from data.seq.base_seq_processor import BaseSeqProcessor
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class TimedInteractionDrecProcessor(BaseDrecProcessor, abc.ABC):
    DATE_COL: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_users(self, interactions):
        item_id_set = set(self.items[self.ITEM_ID_COL].unique())

        interactions = interactions[interactions[self.ITEM_ID_COL].isin(item_id_set)]

        pos_interactions = interactions[interactions[self.LABEL_COL] == 1]

        users = (
            pos_interactions.sort_values([self.USER_ID_COL, self.DATE_COL])
            .groupby(self.USER_ID_COL)[self.ITEM_ID_COL]
            .apply(list)
            .reset_index()
        )

        return users
