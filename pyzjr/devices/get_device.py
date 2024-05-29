import os
import torch

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

def num_workers(batch_size):
    """
    Determine the number of parallel worker processes used for data loading
    """
    return min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

def load_owned_device():
    """
    Return appropriate device objects based on whether the system supports CUDA.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

if __name__=="__main__":
    device = load_owned_device()
    print(device)

