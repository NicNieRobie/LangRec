import json
from typing import cast

import numpy as np

from data.encoding.id_encoder import IDEncoder
from data.encoding.sid_encoder import SIDEncoder


def get_code_embeds(code_path):
    return cast(dict, np.load(code_path, allow_pickle=True).item())


def global_indexer(depth, local_index, num_codes):
    global_index = 0
    for i in range(depth):
        global_index += num_codes[i]
    index = global_index + local_index
    return index


def parse_code_indices(indices):
    num_depth = 0
    num_codes = []

    for iid in indices:
        codes = indices[iid]
        num_depth = max(num_depth, len(codes))
        for i in range(len(num_codes), len(codes)):
            num_codes.append(0)

        for i in range(len(codes)):
            num_codes[i] = max(num_codes[i], codes[i] + 1)

    for iid in indices:
        codes = indices[iid]
        for idx, code in enumerate(codes):
            indices[iid][idx] = global_indexer(idx, code, num_codes)

    return indices, num_codes, sum(num_codes)


def build_code_indices(config, device):
    assert config.code_type in ['id', 'sid'], f'Unknown code type {config.code_type}'

    if config.code_type == 'id':
        encoder = IDEncoder(config, device)
    else:
        encoder = SIDEncoder(config, device)

    result = encoder.encode()

    return {str(k): v for k, v in result.items()}


def get_code_indices(config, device):
    if config.code_path is not None:
        indices = json.load(open(config.code_path))
    elif config.code_type is not None:
        indices = build_code_indices(config, device)
    else:
        raise ValueError('Neither code_path nor code_type were specified when tuning')

    return parse_code_indices(indices)
