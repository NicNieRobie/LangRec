import pandas as pd
import torch
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm

from utils.map import Map


class Dataset(BaseDataset):
    def __init__(self, datalist: pd.DataFrame):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def align(self, batch_size, ascending=False):
        self.datalist = self.datalist.sort_values(Map.LEN_COL, ascending=ascending).reset_index(drop=True)

        print(f'combining dataset by step-wise length alignment')
        num_batches = (len(self.datalist) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), total=num_batches):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, len(self.datalist))
            batch = self.datalist.loc[start_index:end_index - 1]
            max_len = batch[Map.LEN_COL].max()
            self.datalist.loc[start_index:end_index - 1, Map.IPT_COL] = batch[Map.IPT_COL].apply(
                lambda x: list(x)[:max_len] + [0] * (max_len - len(x)))

    def __getitem__(self, idx):
        values = self.datalist.iloc[idx]
        return {column: torch.tensor(values[column], dtype=torch.long) for column in self.datalist.columns}