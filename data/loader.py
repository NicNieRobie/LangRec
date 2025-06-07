import os
import pandas as pd


class Loader:
    def __init__(self, base_dir: str, dataset_name: str, cast_to_string: bool, item_id_col: str, user_id_col: str):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.store_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(self.store_dir, exist_ok=True)

        self.cast_to_string = cast_to_string
        self.item_id_col = item_id_col
        self.user_id_col = user_id_col

    def cast_df(self, df):
        if not self.cast_to_string:
            return df

        if self.item_id_col in df.columns:
            df[self.item_id_col] = df[self.item_id_col].astype(str)

        if self.user_id_col in df.columns:
            df[self.user_id_col] = df[self.user_id_col].astype(str)

        return df

    def load_parquet(self, name: str) -> pd.DataFrame:
        path = os.path.join(self.store_dir, f'{name}.parquet')
        df = pd.read_parquet(path)
        return self.cast_df(df)

    def save_parquet(self, name: str, df: pd.DataFrame):
        path = os.path.join(self.store_dir, f'{name}.parquet')
        df.to_parquet(path)
