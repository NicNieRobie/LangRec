import random
import numpy as np
import torch


def seed(seed_val=4311):
    random.seed(seed_val)

    np.random(seed_val)

    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)

    torch.backends.cudnn.deterministic = True
