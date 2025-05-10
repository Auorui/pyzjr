__version__ = '1.4.14'

import os
import torch
import sys
import platform

def pyzjr_info():
    print("Python版本:", ".".join(map(str, [sys.version_info.major,
                                          sys.version_info.minor,
                                          sys.version_info.micro])))
    print("PyTorch版本:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        print("CUDA设备名称:", torch.cuda.get_device_name(0))
    print("CPU型号:", platform.processor())
    print("CPU核心数量:", os.cpu_count())
    print("系统信息:", platform.system() + " " + platform.version())
    print("---------------------------")
    print("Pyzjr版本:", __version__)

if __name__=="__main__":
    pyzjr_info()