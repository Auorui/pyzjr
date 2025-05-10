"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used for loading devices and releasing device memory.
"""
import gc
import threading
import time
import psutil
import torch
from pyzjr.utils.check import is_parallel

def de_parallel(model):
    """
    将并行模型（DataParallel 或 DistributedDataParallel）转换为单 GPU 模型。
    """
    return model.module if is_parallel(model) else model

def Central_Processing_Unit():
    """
    Returns the CPU device.

    Returns:
        torch.device: The CPU device.
    """
    return torch.device('cpu')

def Graphics_Processing_Unit(i=0, cuda=True):
    """
    If cuda is true and gpu is available, return gpu (i); otherwise, return cpu()
    :param i: Index i, indicating which GPU block to use
    :param cuda: Boolean value indicating whether to use cuda, if not, CPU can be used, set to False
    """
    if cuda and torch.cuda.is_available() and torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def use_all_gpu():
    """
    Those who can use this function must be very rich.
    """
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(i)
    return devices if devices else [torch.device('cpu')]

def load_owned_device(device_ids=0):
    """
    Return appropriate device objects based on whether the system supports CUDA.
    """
    return torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")

def load_device_on_model(model, i=0):
    """
    Load the model onto the specified device, prioritize CUDA devices (if available),
    otherwise load onto the CPU.
    If the number of CUDA devices is greater than 1, use torch. nn DataParallel copies
    the model to all available CUDA devices.
    """
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(f'cuda:{i}')
    elif torch.cuda.is_available():
        model = model.to(f'cuda:{i}')
    else:
        model = model.to('cpu')
    return model

CPU = cpu = Central_Processing_Unit
GPU = gpu = Graphics_Processing_Unit

def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def release_memory(*args):
    """
    Function to release memory resources, particularly useful when working with PyTorch and CUDA.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not isinstance(args, list):
        args = list(args)
    for i in range(len(args)):
        args[i] = None
    gc.collect()
    return args

class PeakCPUMemory:
    def __init__(self):
        self.process = psutil.Process()
        self.peak_monitoring = False

    def peak_monitor(self):
        self.cpu_memory_peak = -1

        while True:
            self.cpu_memory_peak = max(self.process.memory_info().rss, self.cpu_memory_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            if not self.peak_monitoring:
                break

    def start(self):
        self.peak_monitoring = True
        self.thread = threading.Thread(target=self.peak_monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.peak_monitoring = False
        self.thread.join()
        return self.cpu_memory_peak


cpu_peak_tracker = PeakCPUMemory()


def start_measure():
    # Time
    measures = {"time": time.time()}

    gc.collect()
    torch.cuda.empty_cache()

    # CPU mem
    measures["cpu"] = psutil.Process().memory_info().rss
    cpu_peak_tracker.start()

    # GPU mem
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            measures[str(i)] = torch.cuda.memory_allocated(i)
        torch.cuda.reset_peak_memory_stats()

    return measures


def end_measure(start_measures):
    # Time
    measures = {"time": time.time() - start_measures["time"]}

    gc.collect()
    torch.cuda.empty_cache()

    # CPU mem
    measures["cpu"] = (psutil.Process().memory_info().rss - start_measures["cpu"]) / 2**20
    measures["cpu-peak"] = (cpu_peak_tracker.stop() - start_measures["cpu"]) / 2**20

    # GPU mem
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            measures[str(i)] = (torch.cuda.memory_allocated(i) - start_measures[str(i)]) / 2**20
            measures[f"{i}-peak"] = (torch.cuda.max_memory_allocated(i) - start_measures[str(i)]) / 2**20

    return measures


def log_measures(measures, description):
    print(f"{description}:")
    print(f"- Time: {measures['time']:.2f}s")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"- GPU {i} allocated: {measures[str(i)]:.2f}MiB")
            peak = measures[f"{i}-peak"]
            print(f"- GPU {i} peak: {peak:.2f}MiB")
    print(f"- CPU RAM allocated: {measures['cpu']:.2f}MiB")
    print(f"- CPU RAM peak: {measures['cpu-peak']:.2f}MiB")

if __name__ == "__main__":
    device = load_owned_device()
    print(device)

    def simple_computation():
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        c = torch.matmul(a, b)
        return c

    start_measures = start_measure()

    result = simple_computation()
    measures = end_measure(start_measures)
    log_measures(measures, "Simple Computation")