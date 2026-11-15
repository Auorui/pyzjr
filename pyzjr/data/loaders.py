import os
import torch
from torch.utils.data import DataLoader

__all__ = [ "num_worker", "TrainDataloader", "EvalDataloader"]

def num_worker(batch_size):
    """
    Determine the number of parallel worker processes used for data loading
    """
    return min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

class TrainDataloader(DataLoader):
    def __init__(self,
                 train_dataset,
                 batch_size,
                 shuffle=True,
                 pin_memory=None,
                 num_workers=None,
                 drop_last=True,
                 persistent_workers=True,
                 collate_fn=None,
                 **kwargs):
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        actual_workers = num_worker(batch_size) if num_workers is None else num_workers
        actual_persistent = persistent_workers if actual_workers > 0 else False

        super(TrainDataloader, self).__init__(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=actual_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=actual_persistent,
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



