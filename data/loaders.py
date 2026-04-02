import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

__all__ = ["seed_worker", "num_worker", "TrainDataloader", "EvalDataloader"]

def num_worker(batch_size):
    """
    Determine the number of parallel worker processes used for data loading
    """
    return min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

def seed_worker(worker_id):
    # Set dataloader worker seed: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class TrainDataloader(DataLoader):
    def __init__(self,
                 train_dataset,
                 batch_size,
                 shuffle=True,
                 pin_memory=False,
                 num_workers=None,
                 drop_last=True,
                 persistent_workers=True,
                 collate_fn=None,
                 **kwargs):
        super(TrainDataloader, self).__init__(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_worker(batch_size) if num_workers is None else num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            **kwargs
        )


class EvalDataloader(DataLoader):
    def __init__(self,
                 val_dataset,
                 batch_size=1,
                 shuffle=False,
                 pin_memory=False,
                 num_workers=None,
                 persistent_workers=True,
                 collate_fn=None,
                 **kwargs):
        super(EvalDataloader, self).__init__(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_worker(batch_size) if num_workers is None else num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            **kwargs
        )



