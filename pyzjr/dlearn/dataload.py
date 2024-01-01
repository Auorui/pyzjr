
import torch

import random
import numpy as np

def seed_worker(worker_id):
    # Set dataloader worker seed: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)