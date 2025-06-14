import json
import os

from data.encoding.base_encoder import BaseEncoder


class IDEncoder(BaseEncoder):
    DEFAULT_OUTPUT_DIR = 'encoding/'

    def __init__(self, config, device, output_dir=DEFAULT_OUTPUT_DIR):
        super().__init__(config, device, output_dir)

        output_name = f'{self.dataset}_{self.task}_id.json'.lower()
        self.code_output_path = os.path.join(output_dir, output_name)

    def encode(self) -> dict:
        item_vocab = self.processor.item_vocab

        code_dict = dict()
        for item, index in item_vocab.items():
            code_dict[item] = [index]

        os.makedirs(self.output_dir, exist_ok=True)
        json.dump(code_dict, open(self.code_output_path, 'w'), indent=2)

        return code_dict
