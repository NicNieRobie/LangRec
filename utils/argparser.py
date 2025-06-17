import os.path
import sys

from typing import Union
from collections.abc import Iterable

import yaml
import argparse

class ArgParser:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.parser = self._init_parser()

        self.__args_str = None

    @staticmethod
    def _load_config_from_file(config_path: str):
        """
        Loads the config from a single file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'CLI config file not found: {config_path}')

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_config(config_path: Union[Iterable[str], str]):
        """
        Loads the config from a single source file or appends configs from multiple sources
        """
        if isinstance(config_path, Iterable) and not isinstance(config_path, str):
            ans = {"arguments": dict()}
            for path in config_path:
                config = ArgParser._load_config_from_file(path)

                if config.get("description", False):
                    ans["description"] = config.get("description")

                ans["arguments"].update(config["arguments"])

            return ans
        else:
            return ArgParser._load_config_from_file(config_path)

    def _init_parser(self):
        parser = argparse.ArgumentParser(description=self.config.get('description', ''))

        for arg_name, arg_config in self.config.get('arguments', {}).items():
            arg_name_cli = f'--{arg_name}'

            kwargs = {
                'help': arg_config.get('help', ''),
                'required': arg_config.get('required', False)
            }

            arg_type = arg_config.get('type', 'str')

            if arg_type == 'int':
                kwargs['type'] = int
            elif arg_type == 'float':
                kwargs['type'] = float
            elif arg_type == 'bool':
                kwargs['action'] = 'store_true'
            elif arg_type == 'list':
                list_style = arg_config.get('list_style', 'space')
                if list_style == 'comma':
                    kwargs['type'] = lambda s: s.split(',')
                else:
                    kwargs['nargs'] = '+'
                    kwargs['type'] = str

            if 'default' in arg_config and arg_type != 'bool':
                kwargs['default'] = arg_config['default']

            if 'choices' in arg_config:
                kwargs['choices'] = arg_config['choices']

            parser.add_argument(arg_name_cli, **kwargs)

        return parser

    def parse_args(self):
        self.__args_str = ' '.join(sys.argv[1:])

        return self.parser.parse_args()

    @property
    def args_str(self):
        return self.__args_str
