import json
from typing import cast

import numpy as np


def get_code_embeds(code_path):
    return cast(dict, np.load(code_path, allow_pickle=True).item())

def global_indexer(depth, local_index, num_codes):
    global_index = 0
    for i in range(depth):
        global_index += num_codes[i]
    index = global_index + local_index
    return index

def get_code_indices(code_path):
    indices = json.load(open(code_path))

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