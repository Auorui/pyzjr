import platform
import torch
import sys
import os
import pyzjr.version as zjr

def Python_VERSION_INFO():
    return ".".join(map(str, [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]))

def PyTorch_VERSION_INFO():
    return torch.__version__

def CUDA_VERSION_INFO():
    return torch.version.cuda

def GPU_MODEL_INFO():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

def System_VERSION_INFO():
    return platform.system() + " " + platform.version()

def CPU_MODEL_INFO():
    return platform.processor()

def CPU_NUMBERS_COUNT():
    return os.cpu_count()

def Pyzjr_INFO():
    print("Python版本:", Python_VERSION_INFO())
    print("PyTorch版本:", PyTorch_VERSION_INFO())
    if CUDA_VERSION_INFO() and GPU_MODEL_INFO():
        print("CUDA版本:", CUDA_VERSION_INFO())
        print("CUDA设备名称:", GPU_MODEL_INFO())
    print("CPU型号:", CPU_MODEL_INFO())
    print("CPU核心数量:", CPU_NUMBERS_COUNT())
    print("系统信息:", System_VERSION_INFO())
    print("---------------------------")
    print("Pyzjr版本:", zjr.__version__)


if __name__ == "__main__":
    Pyzjr_INFO()