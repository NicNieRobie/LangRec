import os


class Compressor:
    def __init__(self, users, items, store_dir, state, user_id_col, item_id_col, history_col, interactions=None):
        self.users = users
        self.items = items
        self.interactions = interactions
        self.store_dir = store_dir
        self.state = state
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.history_col = history_col

    def compress(self):
        if self.state.compressed:
            return False

        if self.interactions is not None:
            user_set = set(self.interactions[self.user_id_col].unique())
            self.users = self.users[self.users[self.user_id_col].isin(user_set)].drop_duplicates(
                subset=[self.user_id_col]
            )

            item_set = set(self.interactions[self.item_id_col].unique())
        else:
            item_set = set()

        self.users[self.history_col].apply(lambda x: [item_set.add(i) for i in x])
        self.items = self.items[self.items[self.item_id_col].isin(item_set)].reset_index(drop=True)

        if self.interactions is not None:
            self.users.to_parquet(os.path.join(self.store_dir, 'users.parquet'))

        self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))

        self.state.compressed = True
        self.state.write_scores()

        return True
