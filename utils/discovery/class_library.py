import glob
import importlib
import os
from typing import TypeVar, Type, List

from data.base_processor import BaseProcessor
from model.base_model import BaseModel
from metrics.base_metric import BaseMetric

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
            file_name = file_path.split(os.path.sep)[-1].split('.')[0]
            module_path = f'{self._module_dir.replace(os.path.sep, ".")}.{file_name}'

            try:
                module = importlib.import_module(module_path)
                for name, obj in module.__dict__.items():
                    sd = getattr(obj, 'ignore_discovery', False)
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
        return cls.upper().replace(suffix.upper(), '')

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
    def processors():
        return ClassLibraryFactory.create_library(BaseProcessor, 'data', 'processor')

    @staticmethod
    def models():
        return ClassLibraryFactory.create_library(BaseModel, 'model', 'model')

    @staticmethod
    def metrics():
        return ClassLibraryFactory.create_library(BaseMetric, 'metrics')
