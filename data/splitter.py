import os
import random
import pandas as pd


class Splitter:
    def __init__(self, user_id_col, label_col, max_interactions):
        self.user_id_col = user_id_col
        self.label_col = label_col
        self.max_interactions = max_interactions

    @staticmethod
    def _group_iterator(users, interactions):
        for u in users:
            yield interactions.get_group(u)

    def split(self, interactions, users_order, count):
        interactions = interactions.groupby(self.user_id_col)
        iterator = self._group_iterator(users_order, interactions)

        df = pd.DataFrame()
        for group in iterator:
            for label in range(2):
                group_lbl = group[group[self.label_col] == label]
                n = min(self.max_interactions // 2, len(group_lbl))
                df = pd.concat([df, group_lbl.sample(n=n, replace=False)])
            if len(df) >= count:
                break

        return df.reset_index(drop=True)

    def get_user_order(self, interactions, store_dir):
        path = os.path.join(store_dir, 'user_order.txt')
        if os.path.exists(path):
            return [line.strip() for line in open(path)]

        users = interactions[self.user_id_col].unique().tolist()

        random.shuffle(users)

        with open(path, 'w') as f:
            f.write('\n'.join(users))

        return users
