from tqdm import tqdm

from loader.dataset import Dataset
from loader.code_map import CodeMap as Map


class CodeDataset(Dataset):
    def align(self, batch_size, ascending=False):
        self.datalist = self.datalist.sort_values(Map.LEN_COL, ascending=ascending).reset_index(drop=True)

        num_batches = (len(self.datalist) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), total=num_batches, desc="Performing length alignment"):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, len(self.datalist))
            batch = self.datalist.loc[start_index:end_index - 1]
            max_len = batch[Map.LEN_COL].max()
            self.datalist.loc[start_index:end_index - 1, Map.IPT_COL] = batch[Map.IPT_COL].apply(
                lambda x: list(x)[:max_len] + [0] * (max_len - len(x)))
            self.datalist.loc[start_index:end_index - 1, Map.VOC_COL] = batch[Map.VOC_COL].apply(
                lambda x: list(x)[:max_len] + [0] * (max_len - len(x)))