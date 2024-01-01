import os
import torch

def cpu():
    """
    Returns the CPU device.

    Returns:
        torch.device: The CPU device.
    """
    return torch.device('cpu')

def gpu(i=0,cuda=True):
    """
    If cuda is true and gpu is available, return gpu (i); otherwise, return cpu()
    :param i: Index i, indicating which GPU block to use
    :param cuda: Boolean value indicating whether to use cuda, if not, CPU can be used, set to False
    """
    if cuda and torch.cuda.is_available() and torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def allgpu():
    """
    Those who can use this function must be very rich.
    """
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(i)
    return devices if devices else [torch.device('cpu')]

def num_workers(batch_size):
    """
    确定用于数据加载的并行工作进程的数量
    """
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    return num_workers
