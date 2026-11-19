"""Seed all numerical RNGs used by training and standalone evaluation."""
import os
import random

import numpy as np
import torch


def seed_everything(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    # Must precede the first CUDA matrix multiplication in this process.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    """DataLoader supplies a deterministic torch seed from its own generator."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
