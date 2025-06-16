import glob
import importlib
import os
from typing import TypeVar, Type, List

from data.ctr.base_ctr_processor import BaseCTRProcessor
from data.drec.base_drec_processor import BaseDrecProcessor
from data.seq.base_seq_processor import BaseSeqProcessor
from metrics.ctr.base_ctr_metric import BaseCTRMetric
from metrics.seq.base_seq_metric import BaseSeqMetric
from model.ctr.base_model import BaseCTRModel
from model.drec.base_model import BaseDrecModel
from model.seq.base_seq_model import BaseSeqModel

T = TypeVar('T')


class ClassDiscoverer:
    def __init__(self, base_class: Type[T], module_dir: str, suffix: str):
        self._base_class = base_class
        self._module_dir = module_dir
        self._suffix = suffix

    def discover(self):
        suffix_part = f'_{self._suffix}' if self._suffix else ''
        file_pattern = os.path.join(self._module_dir, f'*{suffix_part}.py')
        file_paths = glob.glob(file_pattern)

        classes = []

        for file_path in file_paths:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            module_path = f'{self._module_dir.replace(os.path.sep, ".")}.{file_name}'

            try:
                module = importlib.import_module(module_path)
                for name, obj in module.__dict__.items():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, self._base_class)
                        and obj is not self._base_class
                        and not getattr(obj, 'ignore_discovery', False)
                    ):
                        classes.append(obj)
            except ImportError as e:
                print(f'Error importing module {module_path}: {e}')

        return classes


class ClassRegistry:
    def __init__(self, classes: List[Type[T]], suffix=None, name_normalizer=None):
        self._classes = classes
        self._name_normalizer = name_normalizer or self._default_normalizer
        self._suffix = suffix or ''
        self._class_dict = self._build_class_dict()

    @staticmethod
    def _default_normalizer(cls, suffix):
        pascal_suffix = ''.join(word.capitalize() for word in suffix.split('_'))
        return cls.replace(pascal_suffix, '').upper()

    def _build_class_dict(self):
        class_dict = {}

        for cls in self._classes:
            name = self._name_normalizer(cls.__name__, self._suffix)
            class_dict[name] = cls

        return class_dict

    def get(self, name, default=None):
        return self._class_dict.get(name, default)

    def __getitem__(self, name):
        return self._class_dict[name]

    def __contains__(self, name):
        return name in self._class_dict

    @property
    def class_dict(self):
        return self._class_dict.copy()

    @property
    def classes(self):
        return self._classes.copy()


class ClassLibraryFactory:
    @staticmethod
    def create_library(base_class: Type[T], module_dir: str, suffix: str = '', name_transformer=None):
        discoverer = ClassDiscoverer(base_class, module_dir, suffix)
        classes = discoverer.discover()
        registry = ClassRegistry(classes, suffix, name_transformer)

        library = ClassLibrary(registry)

        return library


class ClassLibrary:
    def __init__(self, registry: ClassRegistry):
        self._registry = registry

    def get_class(self, name, default=None):
        return self._registry.get(name, default)

    def __getitem__(self, name):
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    @property
    def class_dict(self):
        return self._registry.class_dict

    @staticmethod
    def processors(task: str):
        assert task in ["ctr", "seq", "drec"]
        path = os.path.sep.join(["data", task])

        base_class = {
            "ctr": BaseCTRProcessor,
            "seq": BaseSeqProcessor,
            "drec": BaseDrecProcessor,
        }[task]

        return ClassLibraryFactory.create_library(base_class, path, 'processor')

    @staticmethod
    def models(task: str):
        assert task in ["ctr", "seq", "drec"]
        path = os.path.sep.join(['model', task])

        if task == "ctr":
            return ClassLibraryFactory.create_library(BaseCTRModel, path, 'model')
        elif task == "seq":
            return ClassLibraryFactory.create_library(BaseSeqModel, path, 'seq_model')
        else:
            return ClassLibraryFactory.create_library(BaseDrecModel, path, 'model')

    @staticmethod
    def baseline_models(task: str):
        assert task in ["ctr", "seq", "drec"]
        path = os.path.sep.join(['model', task])

        if task == "ctr":
            return ClassLibraryFactory.create_library(BaseCTRModel, path, 'model')
        elif task == "seq":
            return ClassLibraryFactory.create_library(BaseSeqModel, path, 'seq_model')
        else:
            return ClassLibraryFactory.create_library(BaseDrecModel, path, 'model')

    @staticmethod
    def ctr_metrics():
        path = os.path.sep.join(['metrics', 'ctr'])
        return ClassLibraryFactory.create_library(BaseCTRMetric, path)

    @staticmethod
    def seq_metrics():
        path = os.path.sep.join(['metrics', 'seq'])
        return ClassLibraryFactory.create_library(BaseSeqMetric, path)
