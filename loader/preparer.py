import os.path
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.dataset import Dataset
from loader.map import Map
from model.base_model import BaseModel
from data.base_processor import BaseProcessor
from utils.obj_idx_vocabulary import ObjIdxVocabulary


class Preparer:
    DATASET_CLASS = Dataset

    def __init__(self, processor: BaseProcessor, model: BaseModel, config):
        self.processor = processor
        self.model = model
        self.config = config

        self.store_dir = os.path.join(
            'prepare',
            self.get_primary_signature(),
            self.get_secondary_signature(),
        )
        os.makedirs(self.store_dir, exist_ok=True)

        self.iid_vocab = ObjIdxVocabulary(name=Map.IID_COL)
        self.uid_vocab = ObjIdxVocabulary(name=Map.UID_COL)

        self.train_datapath = os.path.join(self.store_dir, 'train.parquet')
        self.valid_datapath = os.path.join(self.store_dir, 'valid.parquet')
        self.test_datapath = os.path.join(self.store_dir, 'test.parquet')

        self.has_generated = (
            (os.path.exists(self.train_datapath) and os.path.exists(self.valid_datapath) or not self.processor.NUM_FINETUNE) and
            (os.path.exists(self.test_datapath) or not self.processor.NUM_TEST)
        )

    def get_primary_signature(self):
        return f'{self.processor.dataset_name}_{self.model.get_name()}_{self.config.task}_{self.config.code_type}'

    def get_secondary_signature(self):
        return f'{self.config.valid_ratio}'

    def tokenize_items(self, source='finetune', item_attrs=None):
        item_set = self.processor.get_item_subset(source, slicer=self.config.history_window)
        item_attrs = item_attrs or self.processor.default_attrs

        item_dict = dict()
        for iid in item_set:
            item_str = self.processor.organize_item(iid, item_attrs)
            item_ids = self.model.generate_simple_input_ids(item_str)
            item_dict[iid] = item_ids[:self.model.max_len // 5]
        return item_dict

    def load_datalist(self, source='finetune'):
        items = self.tokenize_items(source=source)
        line, numbers, user, item, prefix, suffix = self.model.get_special_tokens()

        datalist = []

        max_sequence_len = 0
        print(f'preprocessing on the {self.processor.dataset_name} dataset')
        for index, data in tqdm(
            enumerate(self.processor.generate(slicer=self.config.history_window, source=source, id_only=True)),
            total=len(self.processor.get_source_set(source=source)),
            desc=f"Preprocessing the {self.processor.dataset_name} dataset"
        ):
            uid, iid, history, label = data

            current_item = items[iid][:]
            init_length = len(prefix) + len(user) + len(suffix) + len(item)
            input_ids: Optional[list] = None
            for _ in range(5):
                current_length = init_length + len(current_item)

                idx = len(history) - 1
                while idx >= 0:
                    current_len = len(items[history[idx]]) + len(numbers[len(history) - idx]) + len(line)
                    if current_length + current_len <= self.model.max_len:
                        current_length += current_len
                    else:
                        break
                    idx -= 1

                if idx == len(history) - 1:
                    current_item = current_item[:len(current_item) // 2]
                    continue

                input_ids = prefix + user
                for i in range(idx + 1, len(history)):
                    input_ids += numbers[i - idx] + items[history[i]] + line
                input_ids += item + current_item + suffix
                break

            assert input_ids is not None, f'failed to get input_ids for {index} ({uid}, {iid})'
            max_sequence_len = max(max_sequence_len, len(input_ids))
            datalist.append({Map.IPT_COL: input_ids, Map.LBL_COL: label, Map.UID_COL: uid, Map.IID_COL: iid})

        for data in datalist:
            data[Map.LEN_COL] = len(data[Map.IPT_COL])
            data[Map.IPT_COL] = data[Map.IPT_COL] + [0] * (max_sequence_len - data[Map.LEN_COL])
            data[Map.UID_COL] = self.uid_vocab.append(data[Map.UID_COL])
            data[Map.IID_COL] = self.iid_vocab.append(data[Map.IID_COL])

        print(f'{self.processor.dataset_name} dataset: max_sequence_len: {max_sequence_len}')

        return datalist

    def split_datalist(self, datalist):
        valid_user_set = self.processor.load_valid_user_set(self.config.valid_ratio, self.config.task)
        valid_user_set = [self.uid_vocab[uid] for uid in valid_user_set]

        train_datalist = []
        valid_datalist = []
        for data in datalist:
            if data[Map.UID_COL] in valid_user_set:
                valid_datalist.append(data)
            else:
                train_datalist.append(data)

        return train_datalist, valid_datalist

    def _pack_datalist(self, datalist):
        dataset = self.DATASET_CLASS(datalist)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def load_or_generate(self, mode='train'):
        assert mode in ['train', 'valid', 'test'], f'unknown mode: {mode}'

        if self.has_generated:
            print(f'loading prepared {mode} data on {self.processor.dataset_name} dataset')

            self.iid_vocab.load(self.store_dir)
            self.uid_vocab.load(self.store_dir)

            if mode == 'train':
                return pd.read_parquet(self.train_datapath)
            if mode == 'valid':
                return self._pack_datalist(pd.read_parquet(self.valid_datapath))
            return self._pack_datalist(pd.read_parquet(self.test_datapath))

        if self.processor.finetune_set is not None:
            datalist = self.load_datalist(source='finetune')
            train_datalist, valid_datalist = self.split_datalist(datalist)
            train_datalist = pd.DataFrame(train_datalist)
            valid_datalist = pd.DataFrame(valid_datalist)

            train_datalist.to_parquet(self.train_datapath)
            valid_datalist.to_parquet(self.valid_datapath)
        else:
            train_datalist = valid_datalist = None

        if self.processor.test_set is not None:
            test_datalist = pd.DataFrame(self.load_datalist(source='test'))
            test_datalist.to_parquet(self.test_datapath)
        else:
            test_datalist = None

        self.iid_vocab.save(self.store_dir)
        self.uid_vocab.save(self.store_dir)

        if mode == 'train':
            return train_datalist
        if mode == 'valid':
            return self._pack_datalist(valid_datalist)
        return self._pack_datalist(test_datalist)