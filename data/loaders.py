import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


__all__ = ["RepeatDataLoader", "Repeat_sampler", "seed_worker", "num_worker",
           "TrainDataloader", "EvalDataloader"]

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

class RepeatDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = Repeat_sampler(self.sampler)
        else:
            self.batch_sampler = Repeat_sampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        total_iterations = len(self)
        for i in tqdm(range(total_iterations), desc=f"Iteration", unit="iteration"):
            yield next(self.iterator)

class Repeat_sampler(object):
    """ Function to create a sampler that repeats forever.

    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class TrainDataloader(DataLoader):
    def __init__(self,
                 train_dataset,
                 batch_size,
                 shuffle=True,
                 pin_memory=False,
                 num_workers=None,
                 drop_last=True):
        super(TrainDataloader, self).__init__(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_worker(batch_size) if num_workers is None else num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )


class EvalDataloader(DataLoader):
    def __init__(self,
                 val_dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=None,
                 pin_memory=False):
        super(EvalDataloader, self).__init__(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_worker(batch_size) if num_workers is None else num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )


