import os.path

import yaml
from box import Box


class ProcessorState:
    VERSION = 'v1.0'

    def __init__(self, path):
        self.path = path

        if not os.path.exists(path):
            self.version = self.VERSION
            self.compressed = False
        else:
            data = yaml.safe_load(open(path, 'r'))
            data = Box(data)

            self.version = data.version
            self.compressed = data.compressed

    def write(self):
        data = {
            'compressed': self.compressed,
            'version': self.version
        }

        yaml.dump(data, open(self.path, 'w'))
