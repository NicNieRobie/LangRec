import os
import pickle
from typing import Optional, Union


class ObjIdxVocabulary:
    def __init__(self, name: str):
        self._name = str(name)

        self.o2i, self.i2o = dict(), dict()

        self._editable = True

    @property
    def name(self):
        return self._name

    def append(self, obj, oov_token: Optional[Union[int, str]] = None):
        obj = str(obj)
        if obj not in self.o2i:
            if '\n' in obj:
                raise ValueError(f'token ({obj}) contains line break')

            if not self._editable:
                if oov_token is None:
                    raise ValueError(f'the fixed vocab {self.name} is not allowed to add new token ({obj})')
                if isinstance(oov_token, str):
                    return self[oov_token]
                if len(self) > oov_token >= 0:
                    return oov_token
                raise ValueError(f'oov_token ({oov_token}) is not in the vocab')

            index = len(self)
            self.o2i[obj] = index
            self.i2o[index] = obj

        index = self.o2i[obj]
        return index

    @property
    def size(self):
        return len(self)

    def __len__(self):
        return len(self.i2o)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2o[item]

        return self.o2i[item]

    def __contains__(self, item: str):
        return item in self.o2i

    @property
    def editable(self):
        return self._editable

    def filepath(self, save_dir):
        return os.path.join(save_dir, self.filename)

    @property
    def filename(self):
        return f'{self.name}.vocab'

    def load(self, save_dir: str):
        if not save_dir.endswith('.vocab'):
            save_dir = self.filepath(save_dir)

        self.o2i, self.i2o = {}, {}
        objs = self.load_pkl(save_dir)
        for index, obj in enumerate(objs):
            self.o2i[obj] = index
            self.i2o[index] = obj

        return self

    def save(self, save_dir):
        store_path = self.filepath(save_dir)
        self.save_pkl(list(self), store_path)

        return self

    @staticmethod
    def load_pkl(path: str):
        return pickle.load(open(path, "rb"))

    @staticmethod
    def save_pkl(data: any, path: str):
        with open(path, "wb") as f:
            pickle.dump(data, f)
